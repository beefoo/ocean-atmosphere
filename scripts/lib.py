# -*- coding: utf-8 -*-

import csv
import gzip
import numpy as np
import os
from PIL import Image
from pprint import pprint
import pyopencl as cl
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
        if "." not in string:
            num = int(string)
    except ValueError:
        num = False
        print "Value error: %s" % string
    if num <= -9999 or num >= 9999:
        print "Value unknown: %s" % string
        num = False
    return num

# ATMOSPHERE

# Interpolate between two images using GPU
def getTemperatureImage(dataA, dataB, mu, tRange, gradient):
    # read pixels and floats
    dataA = np.array(dataA)
    dataA = dataA.astype(np.float32)
    dataB = np.array(dataB)
    dataB = dataB.astype(np.float32)
    dataG = np.array(gradient)
    dataG = dataG.astype(np.float32)

    shape = dataA.shape
    h, w, dim = shape

    dataA = dataA.reshape(-1)
    dataB = dataB.reshape(-1)
    dataG = dataG.reshape(-1)

    # the kernel function
    src = """
    __kernel void lerpImage(__global float *a, __global float *b, __global float *grad, __global uchar *result){
        int w = %d;
        int dim = %d;
        int gradLen = %d;
        float mu = %f;
        float minValue = %f;
        float maxValue = %f;

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

        // interpolate the temperature value between a and b dataset
        float temperature = a[i] + mu * (b[i]-a[i]);

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
    """ % (w, dim, len(gradient), mu, tRange[0], tRange[1])

    # Get platforms, both CPU and GPU
    plat = cl.get_platforms()
    CPU = plat[0].get_devices()
    try:
        GPU = plat[1].get_devices()
    except IndexError:
        GPU = "none"
    # Create context for GPU/CPU
    if GPU!= "none":
        print "Using GPU..."
        ctx = cl.Context(GPU)
    else:
        print "Using CPU..."
        ctx = cl.Context(CPU)

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # Kernel function instantiation
    prg = cl.Program(ctx, src).build()

    inA =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dataA)
    inB =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dataB)
    inG =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dataG)
    outResult = cl.Buffer(ctx, mf.WRITE_ONLY, (dataA.astype(np.uint8)).nbytes)

    prg.lerpImage(queue, shape, None , inA, inB, inG, outResult)

    # Copy result
    result = np.empty_like(dataA)
    result = result.astype(np.uint8)
    cl.enqueue_copy(queue, result, outResult)

    result = result.reshape(shape)
    imOut = Image.fromarray(result, mode="RGB")
    return imOut;

def readAtmosphereCSVData(filename):
    print "Reading %s" % filename
    data = []
    with gzip.open(filename, 'rb') as f:
        for line in f:
            row = [triple.split(":") for triple in line.split(",")]
            for i, triple in enumerate(row):
                triple[0] = parseNumber(triple[0]) - 273.15 # temperature: convert kelvin to celsius
                triple[1] = parseNumber(triple[1]) # u vector
                triple[2] = parseNumber(triple[2]) # v vector
                row[i] = tuple(triple)
            data.append(row)
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
