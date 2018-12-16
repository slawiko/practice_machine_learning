import svmutil
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

train_data = svmutil.svm_read_problem('./data/spambase/spambase.train.scaled')
test_data = svmutil.svm_read_problem('./data/spambase/spambase.test.scaled')

def plot(title, x, y, xlabel, ylabel, errorbar=None):
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

  if errorbar:
    plt.errorbar(x, y, errorbar)
  else:
    plt.plot(x, y)
  plt.savefig('./hw8/plots/' + title + '.png')
  plt.gcf().clear()

def get_params(d, C):
  return ['-t', '1', '-d', d, '-c', C, '-q']


def get_k_fold(k, fold, dataset):
  result = np.array_split(dataset, k)
  left = np.concatenate(result[:fold]) if len(result[:fold]) > 0 else np.empty((0,))
  right = np.concatenate(result[fold + 1:]) if len(result[fold + 1:]) > 0 else np.empty((0,))
  return np.concatenate((left, right)), result[fold]

def validation(k, data_x, data_y, d, C):
  losses = []
  params = get_params(d, C)
  print('d = {}, C = {}'.format(d, C))
  for fold in range(k):
    train_x, test_x = get_k_fold(k, fold, data_x)
    train_y, test_y = get_k_fold(k, fold, data_y)
    m = svmutil.svm_train(train_y, train_x, params)
    _, p_acc, __ = svmutil.svm_predict(test_y, test_x, m)
    losses.append(100 - p_acc[0])
  
  return losses

def k_cross_validation(k, data_y, data_x):
  best_loss = 100.0
  Cs = []
  average_losses = []
  stds = []

  Ck = 5
  for d in range(1, 5):
    C = math.pow(2, -Ck)
    for _ in range(Ck * 2 + 1):
      Cs.append(C)
      losses = validation(k, data_x, data_y, d, C)
      stds.append(np.std(losses))
      average_loss = np.average(losses)
      average_losses.append(average_loss)

      if average_loss < best_loss:
        best_loss = average_loss
        best_d = d
        best_C = C
      C *= 2
    
    plot('d={}'.format(d), Cs, average_losses, 'C', 'Loss', stds)

    Cs = []
    average_losses = []
    stds = []
  
  return best_d, best_C

def second(k, data_y, data_x, C):
  ds = range(1, 5)
  average_losses = []
  for d in ds:
    losses = validation(k, data_x, data_y, d, C)
    average_losses.append(np.average(losses))

  plot('C={}'.format(C), ds, average_losses, 'd', 'Loss')

def test(train_y, train_x, test_y, text_x, C):
  ds = range(1, 5)
  svs = []
  losses = []
  for d in ds:
    m = svmutil.svm_train(train_y, train_x, get_params(d, C))
    svs.append(m.get_nr_sv())
    _, p_acc, __ = svmutil.svm_predict(test_y, text_x, m)
    losses.append(100 - p_acc[0])

  plot('test_sv', ds, svs, 'd', 'Support Vectors')
  plot('test_loss', ds, losses, 'd', 'Loss')

k = 10
print('{} cross validation...'.format(k))
d, C = k_cross_validation(k, train_data[0], train_data[1])
print('Cross validation best params: d={}, C={}'.format(d, C))

print('Second part...')
second(k, train_data[0], train_data[1], C)

print('Test...')
test(train_data[0], train_data[1], test_data[0], test_data[1], C)
m = svmutil.svm_train(train_data[0], train_data[1], get_params(d, C))
_, p_acc, __ = svmutil.svm_predict(test_data[0], test_data[1], m)
print('Result accuracy: {}. Support vector count: {}'. format(p_acc[0], m.get_nr_sv()))
