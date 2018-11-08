from reader import split, answers_for_class
from model import Adam, answer
from quality import calculate_accuracy, k_cross_validation_adam

def train_and_test(learning_rate, epoch_cnt, tao, beta1, beta2, train, test, cls):
  train_features, train_answers = split(train)
  train_answers = answers_for_class(train_answers, cls)
  test_features, test_answers = split(test)
  test_answers = answers_for_class(test_answers, cls)
  feature_cnt = len(train_features[0])
  model = Adam(learning_rate=learning_rate, epoch_cnt=epoch_cnt, tao=tao, beta1=beta1, beta2=beta2, n=feature_cnt)
  model.train(train_features, train_answers)
  model_answers = answer(model.predict(test_features))
  accuracy = calculate_accuracy(model_answers, test_answers)

  return accuracy

def run(irises):
  print('### Adam ###')
  for cls in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
    print("{}:".format(cls))
    tao, learning_rate, epoch_cnt, beta1, beta2 = k_cross_validation_adam(5, irises['train'], train_and_test, cls)
    accuracy = train_and_test(learning_rate, epoch_cnt, tao, beta1, beta2, irises['train'], irises['test'], cls)
    print('tao: {}, learning_rate: {}, epoch_cnt: {}, beta1: {}, beta2: {}'.format(tao, learning_rate, epoch_cnt, beta1, beta2))
    print('Accuracy: {:.02f}%\n'.format(accuracy))
