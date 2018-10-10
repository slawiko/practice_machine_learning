features = []

features = sorted(features, key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))

with open('./dist/sorted_features.txt', 'wt') as f:
    for feature in features:
        f.write(str(feature) + ",\n")