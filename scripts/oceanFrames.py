# -*- coding: utf-8 -*-

# python atmosphereFrames.py -debug 1
# ffmpeg -framerate 30/1 -i output/ocean/frame%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p -q:v 1 output/ocean_sample.mp4

import argparse
import datetime
import json
from lib import *
import math
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from netCDF4 import Dataset
import os
from PIL import Image, ImageDraw
from pprint import pprint
import random
import sys

parser = argparse.ArgumentParser()
# Source: https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_third-deg
# Doc: ftp://podaac-ftp.jpl.nasa.gov/allData/oscar/preview/L4/oscar_third_deg/docs/oscarthirdguide.pdf
    # 577,681-point (1201x481) 20 to 420° lon, 80 to -80° lat, 72 measurements per year (~5 day interval)
parser.add_argument('-in', dest="INPUT_FILE", default="../data/raw/oscar_vel2016.nc", help="Input NetCDF file")
parser.add_argument('-intemp', dest="INPUT_TEMPERATURE_FILE", default="../data/raw/sea_surface_temperature/MYD28M_2016-%s.CSV.gz", help="Input CSV temperature files")
parser.add_argument('-out', dest="OUTPUT_FILE", default="../output/ocean/frame%s.png", help="Output image file")
parser.add_argument('-grad', dest="GRADIENT_FILE", default="../data/colorGradientRainbow.json", help="Color gradient json file")
parser.add_argument('-start', dest="DATE_START", default="2016-01-01", help="Date start")
parser.add_argument('-end', dest="DATE_END", default="2016-12-31", help="Date end")
parser.add_argument('-lon', dest="LON_RANGE", default="20,420", help="Longitude range")
parser.add_argument('-lat', dest="LAT_RANGE", default="80,-80", help="Latitude range")
parser.add_argument('-ppp', dest="POINTS_PER_PARTICLE", type=int, default=72, help="Points per particle")
parser.add_argument('-vel', dest="VELOCITY_MULTIPLIER", type=float, default=1.6, help="Number of pixels per degree of lon/lat")
parser.add_argument('-particles', dest="PARTICLES", type=int, default=18000, help="Number of particles to display")
parser.add_argument('-range', dest="TEMPERATURE_RANGE", default="-20.0,40.0", help="Temperature range used for color gradient")
parser.add_argument('-width', dest="WIDTH", type=int, default=2048, help="Target image width")
parser.add_argument('-height', dest="HEIGHT", type=int, default=1024, help="Target image height")
parser.add_argument('-lw', dest="LINE_WIDTH_RANGE", default="1.0,1.0", help="Line width range")
parser.add_argument('-mag', dest="MAGNITUDE_RANGE", default="0.0,1.0", help="Magnitude range")
parser.add_argument('-alpha', dest="ALPHA_RANGE", default="0.0,255.0", help="Alpha range (0-255)")
parser.add_argument('-avg', dest="ROLLING_AVERAGE", type=int, default=6, help="Do a rolling average of x data points")
parser.add_argument('-dur', dest="DURATION", type=int, default=120, help="Duration in seconds")
parser.add_argument('-fps', dest="FPS", type=int, default=30, help="Frames per second")
parser.add_argument('-anim', dest="ANIMATION_DUR", type=int, default=2000, help="How many milliseconds each particle should animate over")
parser.add_argument('-line', dest="LINE_VISIBILITY", type=float, default=0.8, help="Higher = more visible lines")
parser.add_argument('-debug', dest="DEBUG", type=int, default=0, help="If debugging, only output a subset of frames")

args = parser.parse_args()

INPUT_FILE = args.INPUT_FILE
INPUT_TEMPERATURE_FILE = args.INPUT_TEMPERATURE_FILE
OUTPUT_FILE = args.OUTPUT_FILE
GRADIENT_FILE = args.GRADIENT_FILE
DATE_START = [int(d) for d in args.DATE_START.split("-")]
DATE_END = [int(d) for d in args.DATE_END.split("-")]
DURATION = args.DURATION
FPS = args.FPS
DEBUG = args.DEBUG

# Get gradient
GRADIENT = []
with open(GRADIENT_FILE) as f:
    GRADIENT = json.load(f)

dateStart = datetime.date(DATE_START[0], DATE_START[1], DATE_START[2])
dateEnd = datetime.date(DATE_END[0], DATE_END[1], DATE_END[2])

params = {}
params["date_start"] = dateStart
params["date_end"] = dateEnd
params["lon_range"] = [float(d) for d in args.LON_RANGE.strip().split(",")]
params["lat_range"] = [float(d) for d in args.LAT_RANGE.strip().split(",")]
params["points_per_particle"] = args.POINTS_PER_PARTICLE
params["velocity_multiplier"] = args.VELOCITY_MULTIPLIER
params["particles"] = args.PARTICLES
params["temperature_range"] = [float(d) for d in args.TEMPERATURE_RANGE.split(",")]
params["linewidth_range"] = [float(d) for d in args.LINE_WIDTH_RANGE.split(",")]
params["mag_range"] = [float(d) for d in args.MAGNITUDE_RANGE.split(",")]
params["alpha_range"] = [float(d) for d in args.ALPHA_RANGE.split(",")]
params["width"] = args.WIDTH
params["height"] = args.HEIGHT
params["gradient"] = GRADIENT
params["animation_dur"] = args.ANIMATION_DUR
params["rolling_avg"] = args.ROLLING_AVERAGE
params["line_visibility"] = args.LINE_VISIBILITY

# Read temperature data
filenames = [INPUT_TEMPERATURE_FILE % str(month+1).zfill(2) for month in range(12)]

# if debugging, just process 3 seconds
debugFrames = FPS * 3

print "Reading %s files asyncronously..." % len(filenames)
pool = ThreadPool()
tData = pool.map(readOceanCSVData, filenames)
pool.close()
pool.join()
print "Done reading files"

# read the data
ds = Dataset(INPUT_FILE, 'r')

# Extract data from NetCDF file
uData = ds.variables['u'][:]
vData = ds.variables['v'][:]
depth = 0

timeCount = len(uData) # this should be 72, i.e. ~5 day interval
lats = len(uData[0][depth]) # this should be 481
lons = len(uData[0][depth][0]) # this should be 1201;
total = lats * lons
print "%s measurements found with %s degrees (lng) by %s degrees (lat)" % (timeCount, lons, lats)



frames = DURATION * FPS
print "%s frames with duration %s" % (frames, DURATION)

def frameToImage(p):
    # Determine the two vector fields to interpolate from
    print "%s: processing data..." % p["fileOut"]
    data = combineData(p["tData"], p["uData"], p["vData"], tuple(p["lon_range"]), tuple(p["lat_range"]))

    dataCount = len(data)
    dataProgress = p["progress"] * (dataCount - 1)
    dataIndexA0 = int(math.floor(dataProgress))
    dataIndexA1 = dataIndexA0 + p["rolling_avg"]
    dataIndexB0 = int(math.ceil(dataProgress))
    dataIndexB1 = dataIndexB0 + p["rolling_avg"]
    mu = dataProgress - dataIndexA0
    f0 = getWrappedData(data, dataCount, dataIndexA0, dataIndexA1)
    f1 = getWrappedData(data, dataCount, dataIndexB0, dataIndexB1)
    lerpedData = lerpData(f0, f1, mu)

    # Set up temperature background image
    # print "%s: calculating temperature colors" % p["fileOut"]
    baseImage = getTemperatureImage(lerpedData, p)
    baseImage = baseImage.resize((p["width"], p["height"]), resample=Image.BICUBIC)
    # baseImage = baseImage.convert(mode="RGBA")

    # Setup particles
    # print "%s: calculating particles..." % p["fileOut"]
    particles = getParticleData(lerpedData, p)

    print "%s: drawing particles..." % p["fileOut"]
    updatedPx = addParticlesToImage(baseImage, particles, p)
    im = Image.fromarray(updatedPx, mode="RGB")

    im.save(p["fileOut"])
    print "%s: finished." % p["fileOut"]


# Initialize particle starting positions
particleProperties = [
    (random.random(), # a random x
     random.random(), # a random y
     random.random()) # a random offset
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
            "progress": 1.0 * frame / (frames-1),
            "animationProgress": (1.0 * ms / params["animation_dur"]) % 1.0,
            "frame": frame,
            "frames": frames,
            "fileOut": filename,
            "tData": tData,
            "uData": uData,
            "vData": vData,
            "particleProperties": particleProperties
        })
        frameParams.append(p)
    if DEBUG and frame >= debugFrames or DEBUG == 1:
        break

print "Making %s image files asyncronously..." % frames
pool = ThreadPool()
data = pool.map(frameToImage, frameParams)
pool.close()
pool.join()
print "Done."
