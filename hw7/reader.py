import csv
import numpy as np

IRIS_TO_CLASS_MAP = {
  'Iris-setosa': 0,
  'Iris-versicolor': 1,
  'Iris-virginica': 2
}

CLASS_TO_IRIS_MAP = {
  0: 'Iris-setosa',
  1: 'Iris-versicolor',
  2: 'Iris-virginica'
}

def parse_iris(iris):
  result = [float(x) for x in iris[:4]]
  result.append(IRIS_TO_CLASS_MAP[iris[4]])

  return result


def read_irises(path):
  data = []
  with open(path) as csvfile:
    reader = csv.reader(csvfile)
    for iris in reader:
      if len(iris) is 0:
        continue
      data.append(parse_iris(iris))
  
  return np.array(data)

def split(irises):
  features = [iris[:4] for iris in irises]
  answers = [int(iris[4]) for iris in irises]

  return features, answers

def answers_for_class(answers, cls):
  assert cls in IRIS_TO_CLASS_MAP
  return [1 if answer == IRIS_TO_CLASS_MAP[cls] else 0 for answer in answers]
