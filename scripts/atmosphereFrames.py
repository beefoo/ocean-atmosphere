# -*- coding: utf-8 -*-

# python atmosphereFrames.py -debug 1
# ffmpeg -framerate 30/1 -i ../output/atmosphere/frame%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p -q:v 1 ../output/atmosphere_sample.mp4

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
# Source: https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/global-forcast-system-gfs
# Data: https://nomads.ncdc.noaa.gov/data/gfs4/201602/20160229/
# Doc: http://www.nco.ncep.noaa.gov/pmb/docs/on388/tableb.html#GRID4
    # 259920-point (720x361) global Lon/Lat grid. (1,1) at (0E, 90N); matrix layout; prime meridian not duplicated
parser.add_argument('-in', dest="INPUT_FILE", default="../data/raw/atmosphere_100000/gfsanl_4_%s_0000_000.csv.gz", help="Input CSV files")
parser.add_argument('-out', dest="OUTPUT_FILE", default="../output/atmosphere/frame%s.png", help="Output image file")
parser.add_argument('-grad', dest="GRADIENT_FILE", default="../data/colorGradientRainbow.json", help="Color gradient json file")
parser.add_argument('-start', dest="DATE_START", default="2016-01-01", help="Date start")
parser.add_argument('-end', dest="DATE_END", default="2016-12-31", help="Date end")
parser.add_argument('-lon', dest="LON_RANGE", default="0,360", help="Longitude range")
parser.add_argument('-lat', dest="LAT_RANGE", default="90,-90", help="Latitude range")
parser.add_argument('-ppp', dest="POINTS_PER_PARTICLE", type=int, default=72, help="Points per particle")
parser.add_argument('-vel', dest="VELOCITY_MULTIPLIER", type=float, default=0.08, help="Number of pixels per degree of lon/lat")
parser.add_argument('-particles', dest="PARTICLES", type=int, default=12000, help="Number of particles to display")
parser.add_argument('-range', dest="TEMPERATURE_RANGE", default="-20.0,40.0", help="Temperature range used for color gradient")
parser.add_argument('-width', dest="WIDTH", type=int, default=2048, help="Target image width")
parser.add_argument('-height', dest="HEIGHT", type=int, default=1024, help="Target image height")
parser.add_argument('-lw', dest="LINE_WIDTH_RANGE", default="1.0,1.0", help="Line width range")
parser.add_argument('-mag', dest="MAGNITUDE_RANGE", default="0.0,12.0", help="Magnitude range")
parser.add_argument('-alpha', dest="ALPHA_RANGE", default="0.0,255.0", help="Alpha range (0-255)")
parser.add_argument('-avg', dest="ROLLING_AVERAGE", type=int, default=30, help="Do a rolling average of x data points")
parser.add_argument('-dur', dest="DURATION", type=int, default=120, help="Duration in seconds")
parser.add_argument('-fps', dest="FPS", type=int, default=30, help="Frames per second")
parser.add_argument('-anim', dest="ANIMATION_DUR", type=int, default=2000, help="How many milliseconds each particle should animate over")
parser.add_argument('-line', dest="LINE_VISIBILITY", type=float, default=0.5, help="Higher = more visible lines")
parser.add_argument('-debug', dest="DEBUG", type=int, default=0, help="If debugging, only output a subset of frames")

args = parser.parse_args()

INPUT_FILE = args.INPUT_FILE
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

# Read data
date = dateStart
filenames = []
dates = []
while date <= dateEnd:
    filename = INPUT_FILE % date.strftime("%Y%m%d")
    if os.path.isfile(filename):
        filenames.append(filename)
        dates.append(date)
    date += datetime.timedelta(days=1)

# if debugging, just process 3 seconds
debugFrames = FPS * 3
if DEBUG:
    filenames = filenames[:62]
    if DEBUG == 1:
        filenames = filenames[:2]

print "Reading %s files asyncronously..." % len(filenames)
pool = ThreadPool()
data = pool.map(readAtmosphereCSVData, filenames)
pool.close()
pool.join()
print "Done reading files"

lons = len(data[0])
lats = len(data[0][0])
total = lons * lats
print "Lons (%s) x Lats (%s) = %s" % (lons, lats, total)

dateCount = len(dates)
frames = DURATION * FPS
print "%s frames with duration %s" % (frames, DURATION)

def frameToImage(p):
    # Determine the two vector fields to interpolate from
    print "%s: processing data..." % p["fileOut"]
    dataCount = len(p["dates"])
    dataProgress = p["progress"] * (dataCount - 1)
    dataIndexA0 = int(math.floor(dataProgress))
    dataIndexA1 = dataIndexA0 + p["rolling_avg"]
    dataIndexB0 = int(math.ceil(dataProgress))
    dataIndexB1 = dataIndexB0 + p["rolling_avg"]
    mu = dataProgress - dataIndexA0
    f0 = getWrappedData(p["data"], dataCount, dataIndexA0, dataIndexA1)
    f1 = getWrappedData(p["data"], dataCount, dataIndexB0, dataIndexB1)
    lerpedData = lerpData(f0, f1, mu, len(f0[0][0])/2)

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
            "dates": dates,
            "data": data,
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
