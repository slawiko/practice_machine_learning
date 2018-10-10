#!/usr/bin/env python3
import numpy as np
import scipy
import sys
import os
import csv

from numpy import log, sqrt, exp
from feature_calculator import calculate_feature, FEATURE_TO_H_STEP_MAP, FEATURE_TO_W_STEP_MAP, FEATURE_TO_START_MAP
from image_reader import WIDTH, HEIGHT, read_file, CAR, FACE

# features_with_application: [x, y, width, height, type, polarity, [(feature, answer)]]
def calculate_all_features_with_application(images):
    STEP = 1

    results = []
    for t in range(4):
        for x in range(0, HEIGHT, STEP):
            for y in range(0, WIDTH, STEP):
                for width in range(FEATURE_TO_START_MAP[t], WIDTH - y, FEATURE_TO_W_STEP_MAP[t] * STEP):
                    for height in range(FEATURE_TO_START_MAP[t], HEIGHT - x, FEATURE_TO_H_STEP_MAP[t] * STEP):
                        feature_applications_1 = []
                        feature_applications_2 = []
                        for img, answer in images:
                            feature = calculate_feature(img, x, y, width, height, t, 1)
                            if feature is None:
                                continue
                            feature_applications_1.append((int(feature), answer))
                            feature_applications_2.append((-int(feature), answer))
                        results.append([x, y, width, height, t, 1, feature_applications_1])
                        results.append([x, y, width, height, t, -1, feature_applications_2])
    return results

def calculate_features_with_application(images, top_features):
    results = []

    for x, y, width, height, t, polarity in top_features:
        feature_applications = []
        for img, answer in images:
            feature = calculate_feature(img, x, y, width, height, t, polarity)
            if feature is None:
                continue
            feature_applications.append((int(feature), answer))
        results.append([x, y, width, height, t, polarity, feature_applications])
    return results

"""
    H: 
        x <= tetta    car   (CAR)
        x > tetta     face  (FACE)
"""
def calculate_tetta(example):
    t = list(example)
    sorted_features = sorted(t, key=lambda x: x[0][0])
    tetta = sorted_features[0][0][0] - 1
    min_error = 0
    min_error_tetta = tetta
    for [_, answer], weight in sorted_features:
        if answer is CAR:
            min_error += weight
    error = min_error
    m = len(sorted_features)
    for i in range(m):
        feature = sorted_features[i][0][0]
        answer = sorted_features[i][0][1]
        weight = sorted_features[i][1]
        next_feature = sorted_features[i+1][0][0] if i + 1 != m else None
        tetta = feature
        if answer is FACE:
            error += weight
        else:
            error -= weight
        if feature == next_feature:
            continue
        if error < min_error:
            min_error = error
            min_error_tetta = tetta
    return [min_error_tetta, min_error]

def adaboost(features_with_application):
    result = []
    # 6 - because of features_with_application structure
    m = len(features_with_application[0][6])
    D = np.repeat(1 / m, m)

    for example in features_with_application:
        tetta, error = calculate_tetta(zip(example[6], D))
        alpha = 1 / 2 * log((1 - error) / error)
        Z = 2 * sqrt(error * (1 - error))

        result.append([example[0], example[1], example[2], example[3], example[4], example[5], alpha, tetta])
        
        for i in range(m):
            y = example[6][i][1]
            h = CAR if example[6][i][0] <= tetta else FACE
            D[i] = (D[i] * exp(-alpha * y * h)) / Z
    
    result = sorted(result, key=lambda x: x[6], reverse=True)
    return result


folder = sys.argv[-2]
model = sys.argv[-1]

train = []
for fname in os.listdir(os.path.join(folder, 'cars')):
    car = read_file(os.path.join(folder, 'cars', fname))
    train.append([car, CAR])
for fname in os.listdir(os.path.join(folder, 'faces')):
    face = read_file(os.path.join(folder, 'faces', fname))
    train.append([face, FACE])

### Calculates all_features and their alphas ###
# all_features_with_application = calculate_all_features_with_application(train)
# hypotheses = adaboost(all_features_with_application)
# with open('./dist/all_features.txt', 'wt') as f:
#     for hypothesis in hypotheses:
#         f.write(str(hypothesis[:-2]) + ",\n")

top_features = [
    [4, 14, 9, 3, 3, 1],
    [4, 0, 22, 2, 1, 1],
    [4, 18, 3, 3, 3, 1],
    [0, 22, 2, 2, 2, 1],
    [2, 14, 3, 3, 3, 1],
    [8, 18, 9, 3, 3, 1],
    [2, 10, 9, 3, 3, 1],
    [0, 0, 21, 3, 3, 1],
    [4, 20, 3, 3, 3, 1],
    [6, 2, 3, 3, 3, -1],
    [6, 16, 6, 2, 1, 1],
    [16, 0, 15, 3, 3, -1],
    [14, 0, 21, 3, 3, -1],
    [4, 14, 10, 6, 2, 1],
    [2, 22, 2, 10, 2, -1],
    [2, 18, 3, 3, 3, 1],
    [2, 0, 22, 2, 1, 1],
    [14, 20, 3, 3, 3, -1],
    [6, 0, 2, 2, 2, 1],
    [20, 14, 2, 2, 1, 1],
    [0, 18, 10, 10, 2, 1],
    [14, 18, 3, 3, 3, -1],
    [0, 6, 15, 3, 3, 1],
    [0, 0, 26, 6, 2, -1],
    [6, 0, 6, 2, 2, 1],
    [6, 22, 9, 3, 3, 1],
    [4, 16, 3, 3, 3, 1],
    [16, 0, 2, 6, 0, 1],
    [6, 12, 10, 2, 1, 1],
    [8, 18, 15, 5, 3, 1],
    [6, 12, 9, 3, 3, 1],
    [6, 8, 15, 3, 3, 1],
    [18, 2, 6, 2, 1, 1],
    [2, 20, 2, 14, 0, -1],
    [10, 28, 10, 2, 2, 1],
    [2, 2, 21, 3, 3, 1],
    [22, 28, 2, 2, 2, 1],
    [4, 16, 6, 8, 1, 1],
    [4, 16, 22, 14, 2, -1],
    [4, 0, 26, 2, 1, 1],
    [0, 0, 22, 4, 1, 1],
    [0, 4, 2, 10, 2, 1],
    [4, 22, 2, 6, 0, -1],
    [14, 16, 9, 3, 3, -1],
    [4, 2, 3, 3, 3, -1],
    [14, 24, 2, 2, 1, 1],
    [0, 10, 18, 2, 1, -1],
    [0, 18, 3, 5, 3, 1],
    [16, 20, 3, 3, 3, -1],
    [12, 0, 3, 3, 3, 1],
    [2, 4, 14, 2, 1, 1],
    [2, 0, 21, 3, 3, 1],
    [6, 16, 22, 14, 2, -1],
    [2, 16, 3, 3, 3, 1],
    [0, 2, 21, 3, 3, 1],
    [2, 8, 15, 3, 3, 1],
    [4, 10, 15, 3, 3, 1],
    [6, 0, 6, 6, 0, 1],
    [0, 0, 33, 5, 3, -1],
    [12, 0, 2, 2, 1, -1],
    [4, 2, 21, 3, 3, 1],
    [6, 4, 9, 3, 3, -1],
    [8, 24, 9, 3, 3, 1],
    [6, 6, 33, 3, 3, -1],
    [4, 0, 34, 2, 1, 1],
    [20, 18, 2, 2, 2, 1],
    [0, 0, 2, 6, 0, -1],
    [0, 18, 2, 6, 2, 1],
    [10, 12, 2, 6, 0, 1],
    [0, 16, 6, 2, 1, 1],
    [6, 12, 27, 3, 3, -1],
    [8, 14, 15, 3, 3, 1],
    [0, 0, 2, 18, 0, 1],
    [14, 22, 9, 3, 3, 1],
    [0, 20, 3, 3, 3, 1],
    [8, 22, 9, 3, 3, 1],
    [0, 0, 30, 2, 2, 1],
    [0, 16, 3, 3, 3, 1],
    [12, 0, 2, 2, 0, -1],
    [2, 14, 22, 14, 2, -1],
    [6, 12, 26, 2, 1, -1],
    [4, 0, 18, 2, 1, 1],
    [0, 18, 9, 5, 3, 1],
    [20, 20, 2, 2, 2, 1],
    [6, 16, 21, 3, 3, -1],
    [0, 14, 18, 14, 2, -1],
    [0, 14, 6, 6, 2, 1],
    [4, 14, 21, 3, 3, -1],
    [8, 20, 18, 10, 2, -1],
    [0, 14, 3, 3, 3, 1],
    [6, 36, 2, 6, 2, -1],
    [2, 16, 22, 14, 2, -1],
    [4, 16, 22, 18, 2, -1],
    [0, 30, 6, 2, 1, -1],
    [2, 24, 2, 6, 2, 1],
    [2, 0, 26, 2, 1, 1],
    [0, 16, 2, 10, 2, -1],
    [18, 18, 2, 2, 1, 1],
    [6, 26, 3, 5, 3, 1],
    [0, 2, 15, 3, 3, 1]
]
features_with_application = calculate_features_with_application(train, top_features)
hypotheses = adaboost(features_with_application)
with open(model, 'w') as csvfile:
    writer = csv.writer(csvfile)
    for hypothesis in hypotheses:
        writer.writerow(hypothesis)
