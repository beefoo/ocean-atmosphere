# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Interpolate between two images using GPU
def lerpImage(imageA, imageB, mu, outfile):
    imA = Image.open(imageA)
    imB = Image.open(imageB)

    # read pixels and floats
    pxA = np.array(imA)
    pxA = pxA.astype(np.float32)
    pxB = np.array(imB)
    pxB = pxB.astype(np.float32)

    # the kernel function
    kernel = """
    __global__ void lerpImage(float *lerped, float *a, float *b, float mu, int check){
        int i = (threadIdx.x) + blockDim.x * blockIdx.x;
        if(i*3 < check*3) {
            lerped[i*3]= a[i*3] + mu * (b[i*3]-a[i*3]);
            lerped[i*3+1]= a[i*3+1] + mu * (b[i*3+1]-a[i*3+1]);
            lerped[i*3+2]= a[i*3+2] + mu * (b[i*3+2]-a[i*3+2]);
        }
    }
    """

    # define block and grid
    dim = imA.size[0]*imA.size[1]
    checkSize = np.int32(dim)
    BLOCK_SIZE = 1024
    block = (BLOCK_SIZE,1,1)
    grid = (int(dim/BLOCK_SIZE)+1,1,1)

    # Init lerped pixels
    lerpedPx = np.zeros_like(pxA)

    # Compile and get kernel function
    mod = SourceModule(kernel)
    func = mod.get_function("lerpImage")
    func(cuda.Out(lerpedPx), cuda.In(pxA), cuda.In(pxB), np.float32(mu), checkSize, block=block, grid=grid)

    # Convert back to ints and save
    lerpedPx = (np.uint8(lerpedPx))
    imOut = Image.fromarray(lerpedPx, mode="RGB")
    imOut.save(outfile)
    print "Wrote to image file %s" % outfile

lerpImage("imageA.png", "imageB.png", 0.5, "lerpedImage.png")
