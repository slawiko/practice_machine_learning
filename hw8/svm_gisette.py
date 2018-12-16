import liblinearutil
import numpy as np
import math

train_data = liblinearutil.svm_read_problem('./data/gisette/gisette.train.scaled')
test_data = liblinearutil.svm_read_problem('./data/gisette/gisette.test.scaled')

def get_params(s, e, C):
  return ['-s', s, '-e', e, '-c', C, '-q']

def get_k_fold(k, fold, dataset):
  result = np.array_split(dataset, k)
  left = np.concatenate(result[:fold]) if len(result[:fold]) > 0 else np.empty((0,))
  right = np.concatenate(result[fold + 1:]) if len(result[fold + 1:]) > 0 else np.empty((0,))
  return np.concatenate((left, right)), result[fold]

def validation(k, data_x, data_y, s, e, C):
  accuracies = []
  params = get_params(s, e, C)
  print('s = {}, e = {}, C = {}'.format(s, e, C))
  for fold in range(k):
    train_x, test_x = get_k_fold(k, fold, data_x)
    train_y, test_y = get_k_fold(k, fold, data_y)
    m = liblinearutil.train(train_y, train_x, params)
    _, p_acc, __ = liblinearutil.predict(test_y, test_x, m)
    accuracies.append(p_acc[0])
  
  return accuracies

def k_cross_validation(k, data_y, data_x):
  best_acc = 0.0

  Ck = 5
  for s in range(1, 6):
    for e in [0.001, 0.01, 0.1, 0.15]:
      C = math.pow(2, -Ck)
      for _ in range(Ck * 2 + 1):
        accuracies = validation(k, data_x, data_y, s, e, C)
        average_acc = np.average(accuracies)

        if average_acc > best_acc:
          best_acc = average_acc
          best_s = s
          best_e = e
          best_C = C
        C *= 2
  
  return best_s, best_e, best_C

print('Cross validation...')
s, e, C = k_cross_validation(10, train_data[0], train_data[1])
print('Cross validation best params: s={}, e={}, C={}'.format(s, e, C))

print('Train...')
m = liblinearutil.train(train_data[0], train_data[1], get_params(s, e, C))

print('Test...')
_, p_acc, _ = liblinearutil.predict(test_data[0], test_data[1], m)
print('Result accuracy: {}'. format(p_acc[0]))

