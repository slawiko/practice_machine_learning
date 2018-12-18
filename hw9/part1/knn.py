import numpy as np

from reader import read_irises, split
from distances import chebyshev_distance, euklidian_distance, manhattan_distance

def kNN(train_x, train_y, test_x, k, calculate_distance):
  predictions = []
  for instance in test_x:
    distances = []
    for i, neighbor in enumerate(train_x):
      distances.append((calculate_distance(neighbor, instance), train_y[i]))
    k_nearest = sorted(distances, key=lambda x: x[0])[:k]

    votes = {}

    for neighbor in k_nearest:
      if neighbor[1] in votes:
        votes[neighbor[1]] += 1
      else:
        votes[neighbor[1]] = 1

    vote = sorted(votes.items(), key=lambda x: x[1], reverse=True)[0]
    predictions.append(vote[0])
  return predictions

def get_k_fold(k, fold, dataset):
  result = np.array_split(dataset, k)
  left = np.concatenate(result[:fold]) if len(result[:fold]) > 0 else np.empty((0, 5))
  right = np.concatenate(result[fold + 1:]) if len(result[fold + 1:]) > 0 else np.empty((0, 5))
  return np.concatenate((left, right)), result[fold]

def validation(fold_cnt, data, k, distance_func):
  accuracies = []
  for fold in range(fold_cnt):
    train, test = get_k_fold(fold_cnt, fold, data)
    train_x, train_y = split(train)
    test_x, test_y = split(test)
    accuracy = calculate_accuracy(kNN(train_x, train_y, test_x, k, distance_func), test_y)
    accuracies.append(accuracy)
  
  return accuracies

def k_cross_validation(fold_cnt, data):
  best_acc = 0.0

  distances = {
    'euklidian': euklidian_distance,
    'manhattan': manhattan_distance,
    'chebyshev': chebyshev_distance
  }
  for k in range(1, 11):
    for distance_name in distances.keys():
      print('k = {}, {} distance'.format(k, distance_name))
      accuracies = validation(fold_cnt, data, k, distances[distance_name])
      average_acc = np.average(accuracies)
      print('Accuracy: {}\n'.format(average_acc))

      if average_acc > best_acc:
        best_acc = average_acc
        best_k = k
        best_dist = distance_name
  
  print('Best accuracy {} is achieved with k = {}, and {} distance'.format(best_acc, best_k, best_dist))
  return best_acc, best_k, best_dist

def calculate_accuracy(predictions, answers):
  correct = 0
  for i, prediction in enumerate(predictions):
    if prediction == answers[i]:
      correct += 1
  return correct / len(answers)

irises = read_irises('./data/irises/iris.data.csv')

# test_size = 1
# train_size = 5 - test_size
# irises = np.array_split(irises, 5)

# train_data = np.concatenate(irises[:train_size])
# test_data = np.concatenate(irises[train_size:])
# test_features, test_labels = split(test_data)

results = {}

for _ in range(100):
  accuracy, k, distance = k_cross_validation(5, irises)
  key = '{} {}'.format(k, distance)
  if key in results:
    results[key].append(accuracy)
  else:
    results[key] = []
    results[key].append(accuracy)
  np.random.shuffle(irises)

for key in results.keys():
  results[key] = np.average(results[key])

result = sorted(results.items(), key=lambda x: x[1], reverse=True)

print('Best average accuracy {} is achieved with {}'.format(result[0][1], result[0][0]))
print('Second best average accuracy {} is achieved with {}'.format(result[1][1], result[1][0]))