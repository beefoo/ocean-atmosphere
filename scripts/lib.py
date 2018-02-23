# -*- coding: utf-8 -*-

import csv
import gzip
import math
import numpy as np
import numpy.ma as ma
import os
from PIL import Image
from pprint import pprint
import pyopencl as cl
import random
import sys

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

def clamp(value, low=0.0, high=1.0):
    if low > high:
        tmp = low
        low = high
        high = tmp
    value = min(value, high)
    value = max(value, low)
    return value

def getColor(gradient, mu, start=0.0, end=1.0):
    gradientLen = len(gradient)
    start = int(round(start * gradientLen))
    end = int(round(end * gradientLen))
    gradient = gradient[start:end]

    index = int(round(mu * (gradientLen-1)))
    rgb = tuple([int(round(v*255.0)) for v in gradient[index]])
    return rgb

def getWrappedData(data, count, start, end):
    wData = []
    if end <= count:
        wData = data[start:end]
    else: # wrap around to the beginning
        d1 = data[start:count]
        d2 = data[0:(end-count)]
        wData = d1[:] + d2[:]
    return wData

def lerp(a, b, mu):
    return (b-a) * mu + a

def lineIntersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def mean(data):
    n = len(data)
    if n < 1:
        return 0
    else:
        return 1.0 * sum(data) / n

def norm(value, a, b, clamp=True, wrap=False):
    n = 1.0 * (value - a) / (b - a)
    if clamp:
        n = min(n, 1)
        n = max(n, 0)
    if wrap and (n < 0 or n > 1.0):
        n = n % 1.0
    return n

def parseNumber(string):
    num = 0
    try:
        num = float(string)
    except ValueError:
        num = 0.0
        print "Value error: %s" % string
    if num <= -9999 or num >= 9999:
        print "Value unknown: %s" % string
        num = 0.0
    return num

def wrap(value, a, b):
    if value < a:
        value = b + value
    elif value > b:
        value = a + (value - b)
    return value

# ATMOSPHERE

def addParticlesToImage(im, particles, p):
    px = np.array(im)
    px = px.astype(np.uint8)
    shape = px.shape
    h, w, dim = shape

    px = px.reshape(-1)
    particles = particles.reshape(-1)

    # the kernel function
    src = """
    __kernel void addParticles(__global uchar *base, __global uchar *particles, __global uchar *result){
        int w = %d;
        int dim = %d;
        float power = 1.0 - %f; // lower number = more visible lines

        int posx = get_global_id(1);
        int posy = get_global_id(0);
        int i = posy * w * dim + posx * dim;
        int j = posy * w + posx;

        float white = 255.0;
        float alpha = (float) particles[j] / 255.0;
        int r = 0;
        int g = 0;
        int b = 0;

        if (alpha > 0) {
            alpha = pow(alpha*alpha + alpha*alpha, power);
            if (alpha > 1.0) {
                alpha = 1.0;
            }
            float inv = 1.0 - alpha;
            r = (int) round((white * alpha) + ((float) base[i] * inv));
            g = (int) round((white * alpha) + ((float) base[i+1] * inv));
            b = (int) round((white * alpha) + ((float) base[i+2] * inv));
        } else {
            r = base[i];
            g = base[i+1];
            b = base[i+2];
        }

        result[i] = r;
        result[i+1] = g;
        result[i+2] = b;
    }
    """ % (w, dim, p["line_visibility"])

    # Get platforms, both CPU and GPU
    plat = cl.get_platforms()
    GPUs = plat[0].get_devices(device_type=cl.device_type.GPU)
    CPU = plat[0].get_devices()

    # prefer GPUs
    if GPUs and len(GPUs) > 0:
        ctx = cl.Context(devices=GPUs)
    else:
        print "Warning: using CPU"
        ctx = cl.Context(CPU)

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # Kernel function instantiation
    prg = cl.Program(ctx, src).build()

    inA =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=px)
    inB =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=particles)
    outResult = cl.Buffer(ctx, mf.WRITE_ONLY, px.nbytes)

    prg.addParticles(queue, [h, w], None , inA, inB, outResult)

    # Copy result
    result = np.empty_like(px)
    cl.enqueue_copy(queue, result, outResult)

    result = result.reshape(shape)
    return result;

def combineData(tData, uData, vData, uvLonRange, uvLatRange):
    depth = 0
    tLen = len(tData)
    uvLen = len(uData)
    th = len(tData[0])
    tw = len(tData[0][0])
    uvh = len(uData[0][depth])
    uvw = len(uData[0][depth][0])
    uvlon0, uvlon1 = uvLonRange
    uvlat0, uvlat1 = uvLatRange

    tData = np.array(tData, dtype=np.float32)
    uData = np.array(uData)
    vData = np.array(vData)

    # remove nan values
    uData = np.nan_to_num(uData)
    vData = np.nan_to_num(vData)

    uData = uData.astype(np.float32)
    vData = vData.astype(np.float32)

    # convert to 1-dimension
    tData = tData.reshape(-1)
    uData = uData.reshape(-1)
    vData = vData.reshape(-1)

    w = int(uvw * (360.0/(uvlon1-uvlon0)))
    h = w / 2
    dim = 3
    shape = (uvLen, h, w, dim)
    result = np.empty(uvLen * h * w * dim, dtype=np.float32)

    # the kernel function
    src = """
    static float lerpT(float a, float b, float mu) {
        return (b - a) * mu + a;
    }

    static float normLat(float value, float a, float b) {
        return (value - a) / (b - a);
    }

    static float normLon(float value, float a, float b) {
        float mvalue = value;
        if (value < a) {
            mvalue = value + 360.0;
        }
        if (value > b) {
            mvalue = value - 360.0;
        }
        return (mvalue - a) / (b - a);
    }

    __kernel void combineData(__global float *tdata, __global float *udata, __global float *vdata, __global float *result){
        int tLen = %d;
        int uvLen = %d;
        int uvw = %d;
        int uvh = %d;
        int tw = %d;
        int th = %d;
        int w = %d;
        int h = %d;
        int dim = %d;
        float uvlon0 = %f;
        float uvlon1 = %f;
        float uvlat0 = %f;
        float uvlat1 = %f;

        // get current position
        int posx = get_global_id(2);
        int posy = get_global_id(1);
        int post = get_global_id(0);

        // get position in float
        float xf = (float) posx / (float) (w-1);
        float yf = (float) posy / (float) (h-1);
        float tf = (float) post / (float) (uvLen-1);

        // get interpolated temperature
        int tposx = (int) round(xf * (tw-1));
        int tposy = (int) round(yf * (th-1));
        float tpostf = tf * (float) tLen;
        int tposta = (int) floor(tpostf);
        int tpostb = (int) ceil(tpostf);
        if (tpostb >= tLen) { // wrap around to the beginning
            tpostb = 0;
        }
        float tmu = tpostf - floor(tpostf);
        int i0 = tposta * tw * th + tposy * tw + tposx;
        int i1 = tpostb * tw * th + tposy * tw + tposx;
        float tValue = lerpT(tdata[i0], tdata[i1], tmu);

        // convert position from lon 20,420 to -180,180 and lat 80,-80 to 90,-90
        float lat = lerpT(90.0, -90.0, yf);
        float lon = lerpT(-180.0, 180.0, xf);
        float latn = normLat(lat, uvlat0, uvlat1);
        float lonn = normLon(lon, uvlon0, uvlon1);
        float uValue = 0.0;
        float vValue = 0.0;

        // check for invalid latitudes
        if (latn >= 0.0 && latn <= 1.0) {
            int posUVx = (int) round(lonn * (float) (uvw-1));
            int posUVy = (int) round(latn * (float) (uvh-1));
            int uvi = post * uvw * uvh + posUVy * uvw + posUVx;
            uValue = udata[uvi];
            vValue = vdata[uvi];
            if (uValue >= 9999.0 || uValue <= -9999.0) {
                uValue = 0.0;
            }
            if (vValue >= 9999.0 || vValue <= -9999.0) {
                vValue = 0.0;
            }
        }

        int i = post * w * h * dim + posy * w * dim + posx * dim;
        result[i] = tValue;
        result[i+1] = uValue;
        result[i+2] = vValue;
    }
    """ % (tLen, uvLen, uvw, uvh, tw, th, w, h, dim, uvlon0, uvlon1, uvlat0, uvlat1)

    # Get platforms, both CPU and GPU
    plat = cl.get_platforms()
    GPUs = plat[0].get_devices(device_type=cl.device_type.GPU)
    CPU = plat[0].get_devices()

    # prefer GPUs
    if GPUs and len(GPUs) > 0:
        ctx = cl.Context(devices=GPUs)
    else:
        print "Warning: using CPU"
        ctx = cl.Context(CPU)

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # Kernel function instantiation
    prg = cl.Program(ctx, src).build()

    inT =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tData)
    inU =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=uData)
    inV =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vData)
    outResult = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

    prg.combineData(queue, [uvLen, h, w], None , inT, inU, inV, outResult)

    # Copy result
    cl.enqueue_copy(queue, result, outResult)

    result = result.reshape(shape)
    result.astype(float)
    result = result.tolist()

    return result;

# Interpolate between two datasets using GPU
def lerpData(dataA, dataB, mu, offset=0):
    dataLen = len(dataA)
    if dataLen != len(dataB):
        print "Warning: data length mismatch"
    shape = (len(dataA[0]), len(dataA[0][0]), 3)
    h, w, dim = shape
    result = np.empty(h * w * dim, dtype=np.float32)

    # read data as floats
    dataA = np.array(dataA)
    dataA = dataA.astype(np.float32)
    dataB = np.array(dataB)
    dataB = dataB.astype(np.float32)

    # convert to 1-dimension
    dataA = dataA.reshape(-1)
    dataB = dataB.reshape(-1)

    # the kernel function
    src = """
    __kernel void lerpData(__global float *a, __global float *b, __global float *result){
        int dlen = %d;
        int h = %d;
        int w = %d;
        int dim = %d;
        float mu = %f;
        int offsetX = %d;

        // get current position
        int posx = get_global_id(1);
        int posy = get_global_id(0);

        // convert position from 0,360 to -180,180
        int posxOffset = posx;
        if (offsetX > 0 || offsetX < 0) {
            if (posx < offsetX) {
                posxOffset = posxOffset + offsetX;
            } else {
                posxOffset = posxOffset - offsetX;
            }
        }

        // get indices
        int j = posy * w * dim + posx * dim;

        // get the mean values for a and b datasets
        float a1 = 0;
        float a2 = 0;
        float a3 = 0;
        float b1 = 0;
        float b2 = 0;
        float b3 = 0;
        for(int k=0; k<dlen; k++) {
            int i = k * h * w * dim + posy * w * dim + posxOffset * dim;
            a1 = a1 + a[i];
            a2 = a2 + a[i+1];
            a3 = a3 + a[i+2];
            b1 = b1 + b[i];
            b2 = b2 + b[i+1];
            b3 = b3 + b[i+2];
        }
        float denom = (float) dlen;
        a1 = a1 / denom;
        a2 = a2 / denom;
        a3 = a3 / denom;
        b1 = b1 / denom;
        b2 = b2 / denom;
        b3 = b3 / denom;

        // set result
        result[j] = a1 + mu * (b1-a1);
        result[j+1] = a2 + mu * (b2-a2);
        result[j+2] = a3 + mu * (b3-a3);
    }
    """ % (dataLen, h, w, dim, mu, offset)

    # Get platforms, both CPU and GPU
    plat = cl.get_platforms()
    GPUs = plat[0].get_devices(device_type=cl.device_type.GPU)
    CPU = plat[0].get_devices()

    # prefer GPUs
    if GPUs and len(GPUs) > 0:
        ctx = cl.Context(devices=GPUs)
    else:
        print "Warning: using CPU"
        ctx = cl.Context(CPU)

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # Kernel function instantiation
    prg = cl.Program(ctx, src).build()

    inA =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dataA)
    inB =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dataB)
    outResult = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

    prg.lerpData(queue, [h, w], None , inA, inB, outResult)

    # Copy result
    cl.enqueue_copy(queue, result, outResult)

    result = result.reshape(shape)
    return result;

def getParticleData(data, p):
    h = p["particles"]
    w = p["points_per_particle"]
    dim = 4 # four points: x, y, alpha, width

    offset = p["animationProgress"]
    tw = p["width"]
    th = p["height"]
    dh = len(data)
    dw = len(data[0])

    result = np.zeros(tw * th, dtype=np.float32)

    # print "%s x %s x %s = %s" % (w, h, dim, len(result))

    fData = np.array(data)
    fData = fData.astype(np.float32)
    fData = fData.reshape(-1)

    # print "%s x %s x 3 = %s" % (dw, dh, len(fData))

    pData = np.array(p["particleProperties"])
    pData = pData.astype(np.float32)
    pData = pData.reshape(-1)

    # print "%s x 3 = %s" % (h, len(pData))

    # the kernel function
    src = """

    static float lerp(float a, float b, float mu) {
        return (b - a) * mu + a;
    }

    static float det(float a0, float a1, float b0, float b1) {
        return a0 * b1 - a1 * b0;
    }

    static float2 lineIntersection(float x0, float y0, float x1, float y1, float x2, float y2, float x3, float y3) {
        float xd0 = x0 - x1;
        float xd1 = x2 - x3;
        float yd0 = y0 - y1;
        float yd1 = y2 - y3;

        float div = det(xd0, xd1, yd0, yd1);

        float2 intersection;
        intersection.x = -1.0;
        intersection.y = -1.0;

        if (div != 0.0) {
            float d1 = det(x0, y0, x1, y1);
            float d2 = det(x2, y2, x3, y3);
            intersection.x = det(d1, d2, xd0, xd1) / div;
            intersection.y = det(d1, d2, yd0, yd1) / div;
        }

        return intersection;
    }


    static float norm(float value, float a, float b) {
        float n = (value - a) / (b - a);
        if (n > 1.0) {
            n = 1.0;
        }
        if (n < 0.0) {
            n = 0.0;
        }
        return n;
    }

    static float wrap(float value, float a, float b) {
        if (value < a) {
            value = b - (a - value);
        } else if (value > b) {
            value = a + (value - b);
        }
        return value;
    }

    void drawLine(__global float *p, int x0, int y0, int x1, int y1, int w, float alpha);

    void drawLine(__global float *p, int x0, int y0, int x1, int y1, int w, float alpha) {
        int dx = abs(x1-x0);
        int dy = abs(y1-y0);

        if (dx==0 && dy==0) {
            return;
        }

        int sy = 1;
        int sx = 1;
        if (y0>=y1) {
            sy = -1;
        }
        if (x0>=x1) {
            sx = -1;
        }
        int err = dx/2;
        if (dx<=dy) {
            err = -dy/2;
        }
        int e2 = err;

        int x = x0;
        int y = y0;
        for(int i=0; i<w; i++){
            p[y*w+x] = alpha;
            if (x==x1 && y==y1) {
                break;
            }
            e2 = err;
            if (e2 >-dx) {
                err -= dy;
                x += sx;
            }
            if (e2 < dy) {
                err += dx;
                y += sy;
            }
        }
    }

    __kernel void getParticles(__global float *data, __global float *pData, __global float *result){
        int points = %d;
        int dw = %d;
        int dh = %d;
        float tw = %f;
        float th = %f;
        float offset = %f;
        float magMin = %f;
        float magMax = %f;
        float alphaMin = %f;
        float alphaMax = %f;
        float velocityMult = %f;

        // get current position
        int i = get_global_id(0);
        float dx = pData[i*3];
        float dy = pData[i*3+1];
        float doffset = pData[i*3+2];

        // set starting position
        float x = dx * (tw-1);
        float y = dy * (th-1);

        for(int j=0; j<points; j++) {
            // get UV value
            int lon = (int) round(dx * (dw-1));
            int lat = (int) round(dy * (dh-1));
            int dindex = lat * dw * 3 + lon * 3;
            float u = data[dindex+1];
            float v = data[dindex+2];

            // check for invalid values
            if (u >= 9999.0 || u <= -9999.0) {
                u = 0.0;
            }
            if (v >= 9999.0 || v <= -9999.0) {
                v = 0.0;
            }

            // calc magnitude
            float mag = sqrt(u * u + v * v);
            mag = norm(mag, magMin, magMax);

            // determine alpha transparency based on magnitude and offset
            float jp = (float) j / (float) (points-1);
            float progressMultiplier = (jp + offset + doffset) - floor(jp + offset + doffset);
            progressMultiplier = 1.0 - progressMultiplier;
            float alpha = lerp(alphaMin, alphaMax, mag * progressMultiplier);

            float x1 = x + u * velocityMult;
            float y1 = y + (-v) * velocityMult;

            // clamp y
            if (y1 < 0.0) {
                y1 = 0.0;
            }
            if (y1 > (th-1.0)) {
                y1 = th-1.0;
            }

            // check for no movement
            if (x==x1 && y==y1) {
                break;

            // check for invisible line
            } else if (alpha < 1.0) {
                // continue

            // wrap from left to right
            } else if (x1 < 0) {
                float2 intersection = lineIntersection(x, y, x1, y1, (float) 0.0, (float) 0.0, (float) 0.0, th);
                if (intersection.y > 0.0) {
                    drawLine(result, (int) round(x), (int) round(y), 0, (int) intersection.y, (int) tw, round(alpha));
                    drawLine(result, (int) round((float) (tw-1.0) + x1), (int) round(y), (int) (tw-1.0), (int) intersection.y, (int) tw, round(alpha));
                }

            // wrap from right to left
            } else if (x1 > tw-1.0) {
                float2 intersection = lineIntersection(x, y, x1, y1, (float) (tw-1.0), (float) 0.0, (float) (tw-1.0), th);
                if (intersection.y > 0.0) {
                    drawLine(result, (int) round(x), (int) round(y), (int) (tw-1.0), (int) intersection.y, (int) tw, round(alpha));
                    drawLine(result, (int) round((float) x1 - (float)(tw-1.0)), (int) round(y), 0, (int) intersection.y, (int) tw, round(alpha));
                }

            // draw it normally
            } else {
                drawLine(result, (int) round(x), (int) round(y), (int) round(x1), (int) round(y1), (int) tw, round(alpha));
            }

            // wrap x
            x1 = wrap(x1, 0.0, tw-1);
            dx = x1 / tw;
            dy = y1 / th;

            x = x1;
            y = y1;
        }
    }
    """ % (w, dw, dh, tw, th, offset, p["mag_range"][0], p["mag_range"][1], p["alpha_range"][0], p["alpha_range"][1], p["velocity_multiplier"])

    # Get platforms, both CPU and GPU
    plat = cl.get_platforms()
    GPUs = plat[0].get_devices(device_type=cl.device_type.GPU)
    CPU = plat[0].get_devices()

    # prefer GPUs
    if GPUs and len(GPUs) > 0:
        # print "Using GPU"
        ctx = cl.Context(devices=GPUs)
    else:
        print "Warning: using CPU"
        ctx = cl.Context(CPU)

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # Kernel function instantiation
    prg = cl.Program(ctx, src).build()

    inData =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fData)
    inPData =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pData)
    outResult = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

    prg.getParticles(queue, (h, ), None, inData, inPData, outResult)

    # Copy result
    cl.enqueue_copy(queue, result, outResult)

    result = result.reshape((th, tw))
    result = result.astype(np.uint8)

    return result;

# Create image based on temperature data using GPU
def getTemperatureImage(data, p):
    tRange = p["temperature_range"]
    gradient = p["gradient"]

    dataG = np.array(gradient)
    dataG = dataG.astype(np.float32)

    shape = data.shape
    h, w, dim = shape

    data = data.reshape(-1)
    dataG = dataG.reshape(-1)

    # the kernel function
    src = """
    __kernel void lerpImage(__global float *d, __global float *grad, __global uchar *result){
        int w = %d;
        int dim = %d;
        int gradLen = %d;
        float minValue = %f;
        float maxValue = %f;

        // get current position
        int posx = get_global_id(1);
        int posy = get_global_id(0);

        // get index
        int i = posy * w * dim + posx * dim;
        float temperature = d[i];
        int r = 36;
        int g = 36;
        int b = 36;

        // assume large values are invalid
        if (temperature > -9999.0 && temperature < 9999.0) {
            // normalize the temperature
            float norm = (temperature - minValue) / (maxValue - minValue);
            // clamp
            if (norm > 1.0) {
                norm = 1.0;
            }
            if (norm < 0.0) {
                norm = 0.0;
            }
            // get color from gradient
            int gradientIndex = (int) round(norm * (gradLen-1));
            gradientIndex = gradientIndex * 3;
            r = (int) round(grad[gradientIndex] * 255);
            g = (int) round(grad[gradientIndex+1] * 255);
            b = (int) round(grad[gradientIndex+2] * 255);
        }

        // set the color
        result[i] = r;
        result[i+1] = g;
        result[i+2] = b;
    }
    """ % (w, dim, len(gradient), tRange[0], tRange[1])

    # Get platforms, both CPU and GPU
    plat = cl.get_platforms()
    GPUs = plat[0].get_devices(device_type=cl.device_type.GPU)
    CPU = plat[0].get_devices()

    # prefer GPUs
    if GPUs and len(GPUs) > 0:
        # print "Using GPU"
        ctx = cl.Context(devices=GPUs)
    else:
        print "Warning: using CPU"
        ctx = cl.Context(CPU)

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # Kernel function instantiation
    prg = cl.Program(ctx, src).build()

    inData =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    inG =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dataG)
    outResult = cl.Buffer(ctx, mf.WRITE_ONLY, (data.astype(np.uint8)).nbytes)

    prg.lerpImage(queue, [h, w], None, inData, inG, outResult)

    # Copy result
    result = np.empty_like(data)
    result = result.astype(np.uint8)
    cl.enqueue_copy(queue, result, outResult)

    result = result.reshape(shape)
    imOut = Image.fromarray(result, mode="RGB")
    return imOut;

def readAtmosphereCSVData(filename):
    print "Reading %s" % filename
    data = []
    with gzip.open(filename, 'rb') as f:
        rows = list(f)
        rowCount = len(rows)
        data = [None for d in range(rowCount)]
        for i, line in enumerate(rows):
            row = line.split(",")
            for j, triple in enumerate(row):
                triple = triple.split(":")
                triple[0] = parseNumber(triple[0]) - 273.15 # temperature: convert kelvin to celsius
                triple[1] = parseNumber(triple[1]) # u vector
                triple[2] = parseNumber(triple[2]) # v vector
                row[j] = tuple(triple)
            data[i] = row
    print "Done reading %s" % filename
    return data

def readOceanCSVData(filename):
    print "Reading %s" % filename
    data = []
    with gzip.open(filename, 'rb') as f:
        lines = list(f)
        rows = [0 for i in range(len(lines))]
        for i, line in enumerate(lines):
            row = [float(value) for value in line.split(",")]
            rows[i] = row
        data = rows
    return data
