# -*- coding: utf-8 -*-

import csv
import gzip
import sys

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

def readCSVData(filename):
    print "Reading %s" % filename
    data = []
    with gzip.open(filename, 'rb') as f:
        for line in f:
            row = [triple.split(":") for triple in line.split(",")]
            for i, triple in enumerate(row):
                triple[0] = parseNumber(triple[0])
                triple[1] = parseNumber(triple[1])
                triple[2] = parseNumber(triple[2])
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
