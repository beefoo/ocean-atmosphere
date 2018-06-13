# -*- coding: utf-8 -*-

# python generateFrames.py -width 1024 -height 186
# python generateFrames.py -debug 1 -out "../output/metascreen-single%s.png" -width 17280 -height 3240
# caffeinate -i python generateFrames.py -width 17280 -height 3240 -out "/Volumes/youaremyjoy/HoPE/metatest_2018-04-13/frames/frame%s.png"

# python generateFrames.py -in ../data/raw/ocean/oscar_vel2016_20161002.csv.gz -out ../output/perpetual/frame.%s.png -vel 0.3 -ppp 800 -particles 160000 -mag " 0.0,1.0" -line 0.8 -unit Celsius -lon " -180,180" -width 8192 -height 4096 -lw " 4.0,12.0" -latstart 90 -latend 90 -fade 0 -base ../data/bluemarble/world.200410.3x5400x2700.png -grad ../data/colorGradientOcean.json -anim 1000 -fps 30 -debug 1
# ffmpeg -framerate 30/1 -i ../output/perpetual/frame.%03d.png -vf scale=4096:2048 -c:v libx264 -r 30 -pix_fmt yuv420p -q:v 1 ../output/ocean_2018-06-01.mp4

# ffmpeg -framerate 30/1 -i ../output/atmosphere-single/frame%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p -q:v 1 ../output/atmosphere_single_sample.mp4
# ffmpeg -framerate 30/1 -i /Volumes/youaremyjoy/HoPE/metatest_2018-04-13/frames/frame%03d.png -s 1024x186 -c:v libx264 -r 30 -pix_fmt yuv420p -q:v 1 ../output/atmosphere_meta_sample.mp4

# ffmpeg -framerate 29.97/1 -i /Volumes/youaremyjoy/HoPE/metatest_2018-04-13/frames/frame%03d.png -filter:v "crop=1920:1080:1920:0" -c:v libx264 -r 29.97 -pix_fmt yuv420p -q:v 1 ../output/atmosphere_meta_cropped_sample.mp4


import argparse
import datetime
import json
from lib import *
import math
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import os
from PIL import Image, ImageDraw
from pprint import pprint
import random
import sys

parser = argparse.ArgumentParser()

parser.add_argument('-in', dest="INPUT_FILE", default="../data/raw/atmosphere_100000/gfsanl_4_20161002_0000_000.csv.gz", help="Input CSV files")
parser.add_argument('-out', dest="OUTPUT_FILE", default="../output/atmosphere-single/frame%s.png", help="Output image file")
parser.add_argument('-base', dest="BASE_IMAGE", default="../data/earth_base_outline.png", help="Base image file")
parser.add_argument('-grad', dest="GRADIENT_FILE", default="../data/colorGradientRainbow.json", help="Color gradient json file")
parser.add_argument('-lon', dest="LON_RANGE", default="0,360", help="Longitude range")
parser.add_argument('-lat', dest="LAT_RANGE", default="90,-90", help="Latitude range")
parser.add_argument('-ppp', dest="POINTS_PER_PARTICLE", type=int, default=1000, help="Points per particle")
parser.add_argument('-vel', dest="VELOCITY_MULTIPLIER", type=float, default=0.04, help="Number of pixels per degree of lon/lat")
parser.add_argument('-minstep', dest="MIN_STEP", type=float, default=0.0, help="Minimum number of pixel per step")
parser.add_argument('-particles', dest="PARTICLES", type=int, default=60000, help="Number of particles to display")
parser.add_argument('-range', dest="TEMPERATURE_RANGE", default="0.0,38.0", help="Temperature range used for color gradient")
parser.add_argument('-width', dest="WIDTH", type=int, default=2048, help="Target image width")
parser.add_argument('-height', dest="HEIGHT", type=int, default=384, help="Target image height")
parser.add_argument('-lw', dest="LINE_WIDTH_RANGE", default="2.0,2.0", help="Line width range")
parser.add_argument('-lwt', dest="LINE_WIDTH_LAT_RANGE", default="1.0,1.0", help="Line width range based on latitude")
parser.add_argument('-mag', dest="MAGNITUDE_RANGE", default="0.0,12.0", help="Magnitude range")
parser.add_argument('-alpha', dest="ALPHA_RANGE", default="0.0,90.0", help="Alpha range (0-255)")
parser.add_argument('-dur', dest="DURATION", type=int, default=30, help="Duration in seconds")
parser.add_argument('-fps', dest="FPS", type=int, default=29.97, help="Frames per second")
parser.add_argument('-anim', dest="ANIMATION_DUR", type=int, default=4000, help="How many milliseconds each particle should animate over")
parser.add_argument('-line', dest="LINE_VISIBILITY", type=float, default=0.5, help="Higher = more visible lines")
parser.add_argument('-debug', dest="DEBUG", type=int, default=0, help="If debugging, only output a subset of frames")
parser.add_argument('-unit', dest="TEMPERATURE_UNIT", default="Kelvin", help="Temperature unit")
parser.add_argument('-latstart', dest="LAT_START", type=float, default=40.0, help="90 to -90")
parser.add_argument('-latend', dest="LAT_END", type=float, default=35.0, help="90 to -90")
parser.add_argument('-fade', dest="FADE_DURATION", type=int, default=1000, help="Milliseconds to fade in and out")

args = parser.parse_args()

INPUT_FILE = args.INPUT_FILE
OUTPUT_FILE = args.OUTPUT_FILE
GRADIENT_FILE = args.GRADIENT_FILE
DURATION = args.DURATION
FPS = args.FPS
DEBUG = args.DEBUG
TEMPERATURE_UNIT = args.TEMPERATURE_UNIT

# Get gradient
GRADIENT = []
with open(GRADIENT_FILE) as f:
    GRADIENT = json.load(f)

params = {}

params["lon_range"] = [float(d) for d in args.LON_RANGE.strip().split(",")]
params["lat_range"] = [float(d) for d in args.LAT_RANGE.strip().split(",")]
params["points_per_particle"] = args.POINTS_PER_PARTICLE
params["velocity_multiplier"] = args.VELOCITY_MULTIPLIER
params["min_step"] = args.MIN_STEP
params["particles"] = args.PARTICLES
params["temperature_range"] = [float(d) for d in args.TEMPERATURE_RANGE.strip().split(",")]
params["linewidth_range"] = [float(d) for d in args.LINE_WIDTH_RANGE.strip().split(",")]
params["linewidth_lat_range"] = [float(d) for d in args.LINE_WIDTH_LAT_RANGE.strip().split(",")]
params["mag_range"] = [float(d) for d in args.MAGNITUDE_RANGE.strip().split(",")]
params["alpha_range"] = [float(d) for d in args.ALPHA_RANGE.strip().split(",")]
params["width"] = args.WIDTH
params["height"] = args.WIDTH / 2
params["cropped_height"] = args.HEIGHT
params["gradient"] = GRADIENT
params["animation_dur"] = args.ANIMATION_DUR
params["line_visibility"] = args.LINE_VISIBILITY
params["fade_ms"] = args.FADE_DURATION
params["debug"] = (DEBUG > 0)

# crop calculations
latStart = norm(args.LAT_START, 90.0, -90.0)
latEnd = norm(args.LAT_END, 90.0, -90.0)
latHeight = 1.0 * params["cropped_height"] / params["height"]
if (latStart + latHeight) > 1.0:
    latStart = 1.0 - latHeight
if (latEnd + latHeight) > 1.0:
    latEnd = 1.0 - latHeight
params["y_from"] = latStart * params["height"]
params["y_to"] = latEnd * params["height"]

# open and resize base image
print "Reading base image..."
baseImage = Image.open(args.BASE_IMAGE)
baseImage = baseImage.resize((params["width"], params["height"]), resample=Image.BICUBIC)
params["base_image"] = baseImage

# Read data
print "Reading CSV data..."
data = readCSVData({
    "filename": INPUT_FILE,
    "unit": TEMPERATURE_UNIT
})

# Offset the data
print "Offsetting data..."
offset = int(round(params["lon_range"][0] + 180.0))
if offset != 0.0:
    offset = int(round(offset / 360.0 * len(data[0])))
data = offsetData(data, offset)

# Retrieve colors
print "Retrieving color data"
colorImage = getTemperatureImage(data, params)
colorImage = colorImage.resize((params["width"], params["height"]), resample=Image.BICUBIC)
params["color_image"] = colorImage

lats = len(data[0])
lons = len(data[0][0])
total = lons * lats
print "Lons (%s) x Lats (%s) = %s" % (lons, lats, total)

frames = int(round(DURATION * FPS))
print "%s frames with duration %s" % (frames, DURATION)
params["duration_ms"] = DURATION * 1000

# Initialize particle starting positions
particleProperties = [
    (pseudoRandom(i*3), # a stably random x
     pseudoRandom(i*3+1), # a stably random y
     pseudoRandom(i*3+2)) # a stably random offset
    for i in range(params["particles"])
]

frameParams = []
pad = len(str(frames))
for frame in range(frames):
    p = params.copy()
    ms = 1.0 * frame / FPS * 1000
    filename = OUTPUT_FILE % str(frame+1).zfill(pad)
    if not os.path.isfile(filename) or DEBUG >= 1:
        p.update({
            "ms": ms,
            "progress": 1.0 * frame / (frames-1),
            "animationProgress": (1.0 * ms / params["animation_dur"]) % 1.0,
            "frame": frame,
            "frames": frames,
            "fileOut": filename,
            "data": data,
            "particleProperties": particleProperties
        })
        frameParams.append(p)
    if DEBUG:
        break

print "Making %s image files asyncronously..." % frames
pool = ThreadPool(2)
data = pool.map(frameToImage, frameParams)
pool.close()
pool.join()
print "Done."

# print "Making %s image files syncronously..." % frames
# for p in frameParams:
#     frameToImage(p)
