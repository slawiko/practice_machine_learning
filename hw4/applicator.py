#!/usr/bin/env python3
import numpy as np
import scipy
import imageio
import sklearn
import sys
import os
import csv

from feature_calculator import calculate_feature
from image_reader import read_file, CAR, FACE

model_path = sys.argv[-2]
fname = sys.argv[-1]

model = []
with open(model_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        model.append(row)

def hypothesis(feature, tetta):
    if feature <= tetta:
        return CAR
    else:
        return FACE

def apply(model, img):
    answer = 0
    for x, y, width, height, t, polarity, alpha, tetta in model:
        feature = calculate_feature(img, int(x), int(y), int(width), int(height), int(t), int(polarity))
        answer += float(alpha) * hypothesis(int(feature), int(tetta))
    
    if answer <= 0:
        return str(CAR + 1)
    else:
        return str(FACE)

print(apply(model, read_file(fname)))
