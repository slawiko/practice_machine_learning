import numpy as np

def calculate_accuracy(predictions, answers):
  truthy = 0
  total = len(answers)

  for i in range(total):
    if predictions[i] == answers[i]:
      truthy += 1
  
  return truthy / total * 100

# think about new name
def train_test(dataset, k=10):
  test_size = 1
  train_size = k - test_size

  result = np.array_split(dataset, k)

  return { 'train': np.concatenate(result[:train_size]), 'test': np.concatenate(result[train_size:]) }

def k_fold(k, fold, dataset):
  result = np.array_split(dataset, k)
  left = np.concatenate(result[:fold]) if len(result[:fold]) > 0 else np.empty((0, 5))
  right = np.concatenate(result[fold + 1:]) if len(result[fold + 1:]) > 0 else np.empty((0, 5))
  return np.concatenate((left, right)), result[fold]

def k_cross_validation(k, dataset, train_and_test, cls):
  best_accuracy = 0.0
  best_tao = 0.0
  best_lr = 0.0
  best_epoch = 0

  for tao in np.linspace(0, 1, num=5):
    for learning_rate in np.linspace(0.01, 0.05, num=5):
      for epoch_cnt in range(100, 1001, 100):
        accuracies = []
        for fold in range(k):
          train, test = k_fold(k, fold, dataset)
          accuracies.append(train_and_test(learning_rate, epoch_cnt, tao, train, test, cls))

        average_acc = np.average(accuracies)
        if average_acc > best_accuracy:
          best_accuracy = average_acc
          best_tao = tao
          best_lr = learning_rate
          best_epoch = epoch_cnt
  
  return best_tao, best_lr, best_epoch

def k_cross_validation_momentum(k, dataset, train_and_test, cls):
  best_accuracy = 0.0
  best_tao = 0.0
  best_lr = 0.0
  best_epoch = 0
  best_gamma = 0.0

  for tao in np.linspace(0, 1, num=5):
    for learning_rate in np.linspace(0.01, 0.05, num=5):
      for gamma in np.linspace(0, 1, num=10):
        for epoch_cnt in range(100, 1001, 100):
          accuracies = []
          for fold in range(k):
            train, test = k_fold(k, fold, dataset)
            accuracies.append(train_and_test(learning_rate, epoch_cnt, tao, gamma, train, test, cls))

          average_acc = np.average(accuracies)
          if average_acc > best_accuracy:
            best_accuracy = average_acc
            best_tao = tao
            best_lr = learning_rate
            best_epoch = epoch_cnt
            best_gamma = gamma
  
  return best_tao, best_lr, best_epoch, best_gamma

def k_cross_validation_rmsprop(k, dataset, train_and_test, cls):
  best_accuracy = 0.0
  best_tao = 0.0
  best_lr = 0.0
  best_epoch = 0
  best_p = 0

  for tao in np.linspace(0, 1, num=5):
    for learning_rate in np.linspace(0.01, 0.05, num=5):
      for epoch_cnt in range(100, 1001, 100):
        for p in np.linspace(0.7, 0.99, num=5):
          accuracies = []
          for fold in range(k):
            train, test = k_fold(k, fold, dataset)
            accuracies.append(train_and_test(learning_rate, epoch_cnt, tao, p, train, test, cls))

          average_acc = np.average(accuracies)
          if average_acc > best_accuracy:
            best_accuracy = average_acc
            best_tao = tao
            best_lr = learning_rate
            best_epoch = epoch_cnt
            best_p = p
  
  return best_tao, best_lr, best_epoch, best_p


def k_cross_validation_adam(k, dataset, train_and_test, cls):
  best_accuracy = 0.0
  best_tao = 0.0
  best_lr = 0.0
  best_epoch = 0
  best_beta1 = 0
  best_beta2 = 0

  for tao in np.linspace(0, 1, num=5):
    for learning_rate in np.linspace(0.01, 0.05, num=5):
      for epoch_cnt in range(100, 1001, 100):
        for beta1 in np.linspace(0.7, 0.99, num=5):
          for beta2 in np.linspace(0.7, 0.99, num=5):
            accuracies = []
            for fold in range(k):
              train, test = k_fold(k, fold, dataset)
              accuracies.append(train_and_test(learning_rate, epoch_cnt, tao, beta1, beta2, train, test, cls))

            average_acc = np.average(accuracies)
            if average_acc > best_accuracy:
              best_accuracy = average_acc
              best_tao = tao
              best_lr = learning_rate
              best_epoch = epoch_cnt
              best_beta1 = beta1
              best_beta2 = beta2
  
  return best_tao, best_lr, best_epoch, best_beta1, best_beta2