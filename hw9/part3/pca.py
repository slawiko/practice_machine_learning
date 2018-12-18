import datetime

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier


def pca(U, size=None):
  if size == None: # calculate mode, not cut mode
    X = U
    A = np.transpose(X) @ X
    eigenvalues, eigenvectors = np.linalg.eig(A)
    indices = np.argsort(eigenvalues)
    U = np.take(eigenvectors, indices, axis=1)
    W = np.transpose(U)
    return U, W
  U = U[:,-size:]
  W = np.transpose(U)
  return U, W

def kNN(train_x, train_y, test_x, test_y):
  start = datetime.datetime.now()
  nbrs = KNeighborsClassifier(n_neighbors=10)
  nbrs.fit(train_x, train_y)
  accuracy = nbrs.score(test_x, test_y)
  end = datetime.datetime.now()
  print('kNN took {}'.format(end - start))
  return accuracy

def compress(W, X):
  return np.transpose(W @ np.transpose(X))

def restore(U, Y):
  return np.transpose(U @ np.transpose(Y))

def task(X, Y):
  TRAIN_SIZE = 60000
  TEST_SIZE = 10000
  assert TRAIN_SIZE + TEST_SIZE <= len(X)
  train_x = X[:TRAIN_SIZE]
  train_y = Y[:TRAIN_SIZE]
  test_x = X[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
  test_y = Y[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

  accuracy = kNN(train_x, train_y, test_x, test_y)
  print('kNN for clean data accuracy = {}'.format(accuracy))

  U, W = pca(X)
  pcas = [392, 196, 10]
  for size in pcas:
    U, W = pca(U, size)

    compressed_train = compress(W, train_x)
    compressed_test = compress(W, test_x)

    accuracy = kNN(compressed_train, train_y, compressed_test, test_y)
    print('\nkNN for compressed to {} data accuracy = {}'.format(size, accuracy))

    restored_train = restore(U, compressed_train)
    restored_test = restore(U, compressed_test)

    accuracy = kNN(restored_train, train_y, restored_test, test_y)
    print('kNN for restored from {} data accuracy = {}'.format(size, accuracy))
    

mnist = fetch_mldata('MNIST original', data_home='./data/mnist')
data_x = mnist['data']
data_y = mnist['target']

indices = np.random.permutation(len(data_x))
data_x = data_x[indices]
data_y = data_y[indices]

task(data_x, data_y)
