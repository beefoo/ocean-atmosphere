# -*- coding: utf-8 -*-

# Tests whether PyOpenCL is working

import numpy as np
import os
from PIL import Image
from pprint import pprint
import pyopencl as cl
import sys

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

# Interpolate between two images using GPU
def lerpImage(imageA, imageB, mu, outfile):
    imA = Image.open(imageA)
    imB = Image.open(imageB)

    # read pixels and floats
    pxA = np.array(imA)
    pxA = pxA.astype(np.uint8)
    pxB = np.array(imB)
    pxB = pxB.astype(np.uint8)

    shape = pxA.shape
    h, w, dim = shape

    pxA = pxA.reshape(-1)
    pxB = pxB.reshape(-1)

    # the kernel function
    src = """
    __kernel void lerpImage(__global uchar *a, __global uchar *b, __global float *mu, __global uchar *result){
        int w = %d;
        int dim = %d;
        float m = *mu;
        int posx = get_global_id(1);
        int posy = get_global_id(0);
        int i = posy * w * dim + posx * dim;
        result[i] = a[i] + m * (b[i]-a[i]);
        result[i+1] = a[i+1] + m * (b[i+1]-a[i+1]);
        result[i+2] = a[i+2] + m * (b[i+2]-a[i+2]);
    }
    """ % (w, dim)

    # Get platforms, both CPU and GPU
    plat = cl.get_platforms()
    CPU = plat[0].get_devices()
    try:
        GPU = plat[1].get_devices()
    except IndexError:
        GPU = "none"

    # Create context for GPU/CPU
    if GPU!= "none":
        ctx = cl.Context(GPU)
    else:
        ctx = cl.Context(CPU)

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # Kernel function instantiation
    prg = cl.Program(ctx, src).build()

    inA =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pxA)
    inB =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pxB)
    inMu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(mu))
    outResult = cl.Buffer(ctx, mf.WRITE_ONLY, pxA.nbytes)

    prg.lerpImage(queue, shape, None , inA, inB, inMu, outResult)

    # Copy result
    result = np.empty_like(pxA)
    cl.enqueue_copy(queue, result, outResult)

    result = result.reshape(shape)
    imOut = Image.fromarray(result, mode="RGB")
    imOut.save(outfile)
    print "Wrote to image file %s" % outfile

lerpImage("imageA.png", "imageB.png", 0.5, "../output/lerpedImage.png")
