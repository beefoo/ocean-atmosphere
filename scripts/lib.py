# -*- coding: utf-8 -*-

import csv
import gzip
import math
import numpy as np
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
        num = False
        print "Value error: %s" % string
    if num <= -9999 or num >= 9999:
        print "Value unknown: %s" % string
        num = False
    return num

def wrap(value, a, b):
    if value < a:
        value = b + value
    elif value > b:
        value = a + (value - b)
    return value

# ATMOSPHERE

# Interpolate between two datasets using GPU
def lerpData(dataA, dataB, mu):
    # read data as floats
    dataA = np.array(dataA)
    dataA = dataA.astype(np.float32)
    dataB = np.array(dataB)
    dataB = dataB.astype(np.float32)

    shape = dataA.shape
    h, w, dim = shape

    # convert to 1-dimension
    dataA = dataA.reshape(-1)
    dataB = dataB.reshape(-1)

    # the kernel function
    src = """
    __kernel void lerpData(__global float *a, __global float *b, __global float *result){
        int w = %d;
        int dim = %d;
        float mu = %f;

        // get current position
        int posx = get_global_id(1);
        int posy = get_global_id(0);

        // convert position from 0,360 to -180,180
        int halfWidth = w / 2;
        int posxOffset = posx;
        if (posx < halfWidth) {
            posxOffset = posxOffset + halfWidth;
        } else {
            posxOffset = posxOffset - halfWidth;
        }

        // get index
        int i = posy * w * dim + posxOffset * dim;
        int j = posy * w * dim + posx * dim;

        // set result
        result[j] = a[i] + mu * (b[i]-a[i]);
        result[j+1] = a[i+1] + mu * (b[i+1]-a[i+1]);
        result[j+2] = a[i+2] + mu * (b[i+2]-a[i+2]);
    }
    """ % (w, dim, mu)

    # Get platforms, both CPU and GPU
    plat = cl.get_platforms()
    GPUs = plat[0].get_devices(device_type=cl.device_type.GPU)
    CPU = plat[0].get_devices()

    # prefer GPUs
    if GPUs and len(GPUs) > 0:
        ctx = cl.Context(devices=GPUs)
    else:
        ctx = cl.Context(CPU)

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # Kernel function instantiation
    prg = cl.Program(ctx, src).build()

    inA =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dataA)
    inB =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dataB)
    outResult = cl.Buffer(ctx, mf.WRITE_ONLY, dataA.nbytes)

    prg.lerpData(queue, shape, None , inA, inB, outResult)

    # Copy result
    result = np.empty_like(dataA)
    cl.enqueue_copy(queue, result, outResult)

    result = result.reshape(shape)
    return result;

def getParticleData(data, p):
    h = p["particles"]
    w = p["points_per_particle"]
    dim = 4 # four points: x, y, alpha, width

    result = np.empty(h * w * dim, dtype=np.uint8)
    pp = p["particleProperties"]
    offset = p["animationProgress"]
    tw = p["width"]
    th = p["height"]
    dh = len(data)
    dw = len(data[0])

    # the kernel function
    src = """
    static float lerp(float a, float b, float mu) {
        return (b - a) * mu + a;
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

    __kernel void getParticles(__global float *data, __global float *pData, __global uchar *result){
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
        float lineMin = %f;
        float lineMax = %f;
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

            // calc magnitude
            float mag = sqrt(u * u + v * v);
            mag = norm(mag, magMin, magMax);

            // determine alpha transparency and line width based on magnitude and offset
            float jp = (float) j / (float) (points-1);
            float progressMultiplier = (jp + offset + doffset) - floor(jp + offset + doffset);
            float alpha = lerp(alphaMin, alphaMax, mag * progressMultiplier);
            float linewidth = lerp(lineMin, lineMax, mag * progressMultiplier);

            int pindex = i * points * 4 + j * 4;
            result[pindex] = (int) round(x);
            result[pindex+1] = (int) round(y);
            result[pindex+2] = (int) round(alpha);
            result[pindex+3] = (int) round(linewidth*1000);

            x = x + u * velocityMult;
            y = y + (-v) * velocityMult;

            if (y < 0.0) {
                y = 0.0;
            }
            if (y > (th-1.0)) {
                y = th-1.0;
            }
            x = wrap(x, 0.0, tw-1);
            dx = x / tw;
            dy = y / th;
        }
    }
    """ % (w, dw, dh, tw, th, offset, p["mag_range"][0], p["mag_range"][1], p["alpha_range"][0], p["alpha_range"][1], p["linewidth_range"][0], p["linewidth_range"][1], p["velocity_multiplier"])

    # Get platforms, both CPU and GPU
    plat = cl.get_platforms()
    GPUs = plat[0].get_devices(device_type=cl.device_type.GPU)
    CPU = plat[0].get_devices()

    # prefer GPUs
    if GPUs and len(GPUs) > 0:
        print "Using GPU"
        ctx = cl.Context(devices=GPUs)
    else:
        print "Using CPU"
        ctx = cl.Context(CPU)

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # Kernel function instantiation
    prg = cl.Program(ctx, src).build()

    data = np.array(data)
    data = data.astype(np.float32)
    data = data.reshape(-1)

    pData = np.array(pp)
    pData = pData.astype(np.float32)
    pData = pData.reshape(-1)

    inData =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    inPData =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pData)
    outResult = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

    prg.getParticles(queue, (h, ), None, inData, inPData, outResult)

    # Copy result
    cl.enqueue_copy(queue, result, outResult)

    result = result.reshape((h, w, dim))
    result = result.astype(int)

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

        // set the color
        i = posy * w * dim + posx * dim;
        result[i] = (int) round(grad[gradientIndex] * 255);
        result[i+1] = (int) round(grad[gradientIndex+1] * 255);
        result[i+2] = (int) round(grad[gradientIndex+2] * 255);
    }
    """ % (w, dim, len(gradient), tRange[0], tRange[1])

    # Get platforms, both CPU and GPU
    plat = cl.get_platforms()
    GPUs = plat[0].get_devices(device_type=cl.device_type.GPU)
    CPU = plat[0].get_devices()

    # prefer GPUs
    if GPUs and len(GPUs) > 0:
        print "Using GPU"
        ctx = cl.Context(devices=GPUs)
    else:
        print "Using CPU"
        ctx = cl.Context(CPU)

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # Kernel function instantiation
    prg = cl.Program(ctx, src).build()

    inData =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    inG =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dataG)
    outResult = cl.Buffer(ctx, mf.WRITE_ONLY, (data.astype(np.uint8)).nbytes)

    prg.lerpImage(queue, shape, None, inData, inG, outResult)

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

class Particle:

    def __init__(self):
        self.startPosition = ()
        self.boundX = ()
        self.boundY = ()
        self.positions = []

    def loadFromField(self, field):
        print "TODO"
