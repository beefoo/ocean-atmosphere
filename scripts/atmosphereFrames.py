# -*- coding: utf-8 -*-

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
parser.add_argument('-vel', dest="VELOCITY_MULTIPLIER", type=float, default=0.06, help="Velocity mulitplier")
parser.add_argument('-particles', dest="PARTICLES", type=int, default=6000, help="Number of particles to display")
parser.add_argument('-range', dest="TEMPERATURE_RANGE", default="-20.0,40.0", help="Temperature range used for color gradient")
parser.add_argument('-width', dest="WIDTH", type=int, default=2048, help="Target image width")
parser.add_argument('-height', dest="HEIGHT", type=int, default=1024, help="Target image height")
parser.add_argument('-dur', dest="DURATION", type=int, default=120, help="Duration in seconds")
parser.add_argument('-fps', dest="FPS", type=int, default=30, help="Frames per second")

args = parser.parse_args()

INPUT_FILE = args.INPUT_FILE
OUTPUT_FILE = args.OUTPUT_FILE
GRADIENT_FILE = args.GRADIENT_FILE
DATE_START = [int(d) for d in args.DATE_START.split("-")]
DATE_END = [int(d) for d in args.DATE_END.split("-")]
DURATION = args.DURATION
FPS = args.FPS

# Get gradient
GRADIENT = []
with open(GRADIENT_FILE) as f:
    GRADIENT = json.load(f)

params = {}
params["lon_range"] = [float(d) for d in args.LON_RANGE.strip().split(",")]
params["lat_range"] = [float(d) for d in args.LAT_RANGE.strip().split(",")]
params["points_per_particle"] = args.POINTS_PER_PARTICLE
params["velocity_multiplier"] = args.VELOCITY_MULTIPLIER
params["particles"] = args.PARTICLES
params["temperature_range"] = [float(d) for d in args.TEMPERATURE_RANGE.split(",")]
params["width"] = args.WIDTH
params["height"] = args.HEIGHT
params["gradient"] = GRADIENT

# Read data
dateStart = datetime.date(DATE_START[0], DATE_START[1], DATE_START[2])
dateEnd = datetime.date(DATE_END[0], DATE_END[1], DATE_END[2])
date = dateStart

filenames = []
dates = []
while date <= dateEnd:
    filename = INPUT_FILE % date.strftime("%Y%m%d")
    if os.path.isfile(filename):
        filenames.append(filename)
        dates.append(date)
    date += datetime.timedelta(days=1)

print "Reading %s files asyncronously..." % len(filenames)
pool = ThreadPool()
data = pool.map(readCSVData, filenames)
pool.close()
pool.join()
print "Done reading files"

lons = len(data[0][0][0])
lats = len(data[0][0])
total = lons * lats
print "Lons (%s) x Lats (%s) = %s" % (lons, lats, total)

dateCount = len(dates)
frames = DURATION * FPS
print "%s frames with duration %s" % (frames, DURATION)

def frameToImage(p):
    print "Processing %s" % p["fileOut"]

    # Determine the two vector fields to interpolate from

    # Interpolate between two fields

    # Set up temperature background image
    # Interpolate between two temperature images

    # Setup particles

    # Draw particles

    print "Finished %s" % p["fileOut"]


# Initialize particle starting positions
particleStartingPositions = [(random.uniform(params["lon_range"][0], params["lon_range"][1]), random.uniform(params["lat_range"][1], params["lat_range"][0])) for i in range(params["particles"])]

# Initialize particles offsets
particleOffsets = [random.random() for i in range(params["particles"])]

frameParams = []
pad = len(str(len(frames)))
for frame in range(frames):
    p = params.copy()
    p.update({
        "frame": frame,
        "frames": frames,
        "fileOut": OUTPUT_FILE % str(frame+1).zfill(pad),
        "dates": dates,
        "data": data,
        "particleStartingPositions": particleStartingPositions,
        "particleOffsets": particleOffsets
    })
    frameParams.append(p)

print "Making %s image files asyncronously..." % frames
pool = ThreadPool()
data = pool.map(frameToImage, frameParams)
pool.close()
pool.join()
print "Done."
