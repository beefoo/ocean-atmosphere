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
parser.add_argument('-vel', dest="VELOCITY_MULTIPLIER", type=float, default=0.08, help="Number of pixels per degree of lon/lat")
parser.add_argument('-particles', dest="PARTICLES", type=int, default=10000, help="Number of particles to display")
parser.add_argument('-range', dest="TEMPERATURE_RANGE", default="-20.0,40.0", help="Temperature range used for color gradient")
parser.add_argument('-width', dest="WIDTH", type=int, default=2048, help="Target image width")
parser.add_argument('-height', dest="HEIGHT", type=int, default=1024, help="Target image height")
parser.add_argument('-lw', dest="LINE_WIDTH_RANGE", default="0.2,2.4", help="Line width range")
parser.add_argument('-mag', dest="MAGNITUDE_RANGE", default="0.0,12.0", help="Magnitude range")
parser.add_argument('-alpha', dest="ALPHA_RANGE", default="0.0,255.0", help="Alpha range")
parser.add_argument('-dur', dest="DURATION", type=int, default=120, help="Duration in seconds")
parser.add_argument('-fps', dest="FPS", type=int, default=30, help="Frames per second")
parser.add_argument('-anim', dest="ANIMATION_DUR", type=int, default=3000, help="How many milliseconds each particle should animate over")

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

print "Reading %s files asyncronously..." % len(filenames)
pool = ThreadPool()
data = pool.map(readAtmosphereCSVData, filenames[:2])
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
    delta = p["date_end"] - p["date_start"]
    dayProgress = p["progress"] * (delta.days + 1.0)
    dateTarget = p["date_start"] + datetime.timedelta(days=int(dayProgress))
    i0 = len(p["dates"])-1
    i1 = 0
    for i, date in enumerate(p["dates"]):
        if date == dateTarget:
            i0 = i
            i1 = i + 1
            break
        elif date > dateTarget:
            i0 = i - 1
            i1 = i
            break
    if i0 < 0:
        i0 = len(p["dates"])-1
    if i1 >= len(p["dates"]):
        i1 = 0
    d0 = (p["dates"][i0]-p["date_start"]).days
    d1 = (p["dates"][i1]-p["date_start"]).days
    if d1 > d0 and d0 <= dayProgress <= d1:
        mu = norm(dayProgress, d0, d1)
    else:
        mu = dayProgress % 1.0
    f0 = p["data"][i0]
    f1 = p["data"][i1]

    lerpedData = lerpData(f0, f1, mu)

    # Set up temperature background image
    print "%s: calculating temperature colors" % p["fileOut"]
    baseImage = getTemperatureImage(lerpedData, p)
    baseImage = baseImage.resize((p["width"], p["height"]), resample=Image.BICUBIC)
    # baseImage = baseImage.convert(mode="RGBA")

    # Setup particles
    print "%s: calculating particles..." % p["fileOut"]
    particles = getParticleData(lerpedData, p)

    # Draw particles
    print "%s: drawing particles..." % p["fileOut"]
    draw = ImageDraw.Draw(baseImage, 'RGBA')
    w = p["width"]
    h = p["height"]
    hw = w/2

    for particle in particles:
        for i, point in enumerate(particle):
            if i > 0:
                prev = particle[i-1]
                pwidth = max(1, int(round(point[3]/1000.0)));
                # going from right side back to the left side
                if prev[0]-point[0] > hw:
                    intersection = lineIntersection(
                        ((prev[0], prev[1]), (point[0]+w-1, point[1])),
                        ((w-1, 0), (w-1, h))
                    )
                    if intersection:
                        draw.line([prev[0], prev[1], intersection[0], intersection[1]], fill=(255, 255, 255, point[2]), width=pwidth)
                        draw.line([0, intersection[1], point[0], point[1]], fill=(255, 255, 255, point[2]), width=pwidth)
                    # else:


                # going from left side around to the right side
                elif point[0]-prev[0] > hw:
                    intersection = lineIntersection(
                        ((prev[0], prev[1]), (point[0]-w-1, point[1])),
                        ((0, 0), (0, h))
                    )
                    if intersection:
                        draw.line([prev[0], prev[1], intersection[0], intersection[1]], fill=(255, 255, 255, point[2]), width=pwidth)
                        draw.line([w-1, intersection[1], point[0], point[1]], fill=(255, 255, 255, point[2]), width=pwidth)
                    # else:

                # draw line normally
                else:
                    draw.line([prev[0], prev[1], point[0], point[1]], fill=(255, 255, 255, point[2]), width=pwidth)
    del draw

    baseImage.save(p["fileOut"])
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
    p.update({
        "progress": 1.0 * frame / (frames-1),
        "animationProgress": (1.0 * ms / params["animation_dur"]) % 1.0,
        "frame": frame,
        "frames": frames,
        "fileOut": OUTPUT_FILE % str(frame+1).zfill(pad),
        "dates": dates,
        "data": data,
        "particleProperties": particleProperties
    })
    frameParams.append(p)
    break

print "Making %s image files asyncronously..." % frames
pool = ThreadPool()
data = pool.map(frameToImage, frameParams)
pool.close()
pool.join()
print "Done."
