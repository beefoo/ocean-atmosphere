# -*- coding: utf-8 -*-

import argparse
import math
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import os
from PIL import Image
from pprint import pprint
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-in', dest="INPUT_FILE", default="../output/atmosphere/frame%s.png,../output/ocean/frame%s.png", help="Input image files")
parser.add_argument('-out', dest="OUTPUT_FILE", default="../output/composite/frame%s.png", help="Output image file")
parser.add_argument('-frames', dest="FRAMES", type=int, default=3600, help="Number of frames")
parser.add_argument('-w', dest="WIDTH", type=int, default=2048, help="Target width")
parser.add_argument('-h', dest="HEIGHT", type=int, default=2048, help="Target height")

args = parser.parse_args()

INPUT_FILE = args.INPUT_FILE.split(",")
OUTPUT_FILE = args.OUTPUT_FILE
FRAMES = args.FRAMES
WIDTH = args.WIDTH
HEIGHT = args.HEIGHT

pad = len(str(FRAMES))
params = []
for f in range(FRAMES):
    frame = (f+1).zfill(pad)
    fnames = [ff % frame for ff in INPUT_FILE]
    params.append({
        "w": WIDTH,
        "h": HEIGHT,
        "fnames": fnames,
        "fout": OUTPUT_FILE % frame
    })

def compositeFiles(p):
    fname1 = p["fnames"][0]
    fname2 = p["fnames"][1]
    w = p["w"]
    h = p["h"]
    hh = h / 2

    im1 = Image.open(fname1)
    im2 = Image.open(fname2)

    comp = Image.new('RGB', (p["w"], p["h"]))
    comp.paste(im1)
    comp.paste(im2, box=(0, hh))
    comp.save(p["fout"])

    print "Saved %s" % p["fout"]

print "Processing %s frames asyncronously..." % len(params)
pool = ThreadPool()
data = pool.map(compositeFiles, params)
pool.close()
pool.join()
print "Done."