import numpy as np

def minkowski_distance(x1, x2, p):
  assert len(x1), len(x2)
  return np.linalg.norm(x1 - x2, p)

def manhattan_distance(x1, x2):
  return minkowski_distance(x1, x2, 1)

def euklidian_distance(x1,  x2):
  return minkowski_distance(x1, x2, 2)

def chebyshev_distance(x1, x2):
  return minkowski_distance(x1, x2, np.inf)