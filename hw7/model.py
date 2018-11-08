import numpy as np

def sigmoid(t):
  return 1 / (1 + np.exp(-t))

def answer(predictions):
  return [1 if prediction > 0.5 else 0 for prediction in predictions]

class SGD:
  def __init__(self, learning_rate, epoch_cnt, tao, n):
    self.learning_rate = learning_rate
    self.epoch_cnt = epoch_cnt
    self.tao = tao
    self.w = np.zeros(n)
    self.b = 0

  def predict(self, x):
    return sigmoid(np.dot(x, self.w) + self.b)

  def train(self, features, answers):
    for i in range(self.epoch_cnt):
      index = i % len(features)
      answer = answers[index]
      x = features[index]
      prediction = self.predict(x)
      self.w = self.w - self.learning_rate * ((prediction - answer) * x + 2 * self.tao * self.w)
      self.b = self.b - self.learning_rate * ((prediction - answer) + 2 * self.tao * self.b)


class BGD:
  def __init__(self, learning_rate, epoch_cnt, tao, n):
    self.learning_rate = learning_rate
    self.epoch_cnt = epoch_cnt
    self.tao = tao
    self.w = np.zeros(n)
    self.b = 0

  def predict(self, x):
    return sigmoid(np.dot(x, self.w) + self.b)

  def train(self, features, answers):
    m = len(features)
    for _ in range(self.epoch_cnt):
      predictions = self.predict(features)
      self.w = self.w - self.learning_rate * ((np.dot(predictions - answers, features) / m) + 2 * self.tao * self.w)
      self.b = self.b - self.learning_rate * (np.sum(predictions - answers) / m + 2 * self.tao * self.b)


class SGDMomentum:
  def __init__(self, learning_rate, epoch_cnt, tao, gamma, n):
    self.learning_rate = learning_rate
    self.epoch_cnt = epoch_cnt
    self.tao = tao
    self.gamma = gamma
    self.n = n
    self.w = np.zeros(self.n)
    self.b = 0

  def predict(self, x):
    return sigmoid(np.dot(x, self.w) + self.b)

  def train(self, features, answers):
    uw = np.zeros(self.n)
    ub = 0
    for i in range(self.epoch_cnt):
      index = i % len(features)
      answer = answers[index]
      x = features[index]
      prediction = self.predict(x)
      uw = self.gamma * uw + self.learning_rate * ((prediction - answer) * x + 2 * self.tao * self.w)
      ub = self.gamma * ub + self.learning_rate * ((prediction - answer) + 2 * self.tao * self.b)
      self.w = self.w - uw
      self.b = self.b - ub

class SGDNesterovMomentum:
  def __init__(self, learning_rate, epoch_cnt, tao, gamma, n):
    self.learning_rate = learning_rate
    self.epoch_cnt = epoch_cnt
    self.tao = tao
    self.gamma = gamma
    self.n = n
    self.w = np.zeros(self.n)
    self.u = np.zeros(self.n)
    self.b = 0

  def predict(self, x):
    return sigmoid(np.dot(x, self.w) + self.b)

  def predict_nesterov(self, x):
    return sigmoid(np.dot(x, (self.w - self.gamma * self.u)) + self.b)

  def train(self, features, answers):
    ub = 0
    for i in range(self.epoch_cnt):
      index = i % len(features)
      answer = answers[index]
      x = features[index]
      prediction = self.predict_nesterov(x)
      self.u = self.gamma * self.u + self.learning_rate * ((prediction - answer) * x + 2 * self.tao * self.w)
      ub = self.gamma * ub + self.learning_rate * ((prediction - answer) + 2 * self.tao * self.b)
      self.w = self.w - self.u
      self.b = self.b - ub

class Adagrad:
  def __init__(self, learning_rate, epoch_cnt, tao, n):
    self.learning_rate = learning_rate
    self.epoch_cnt = epoch_cnt
    self.tao = tao
    self.n = n
    self.w = np.zeros(n)
    self.b = 0

  def predict(self, x):
    return sigmoid(np.dot(x, self.w) + self.b)

  def train(self, features, answers):
    G = np.zeros(self.n)
    Gb = 0
    eps = 1e-8
    for i in range(self.epoch_cnt):
      index = i % len(features)
      answer = answers[index]
      x = features[index]
      prediction = self.predict(x)
      g = (prediction - answer) * x + 2 * self.tao * self.w
      gb = (prediction - answer) + 2 * self.tao * self.b
      G = G + np.square(g)
      Gb = Gb + np.square(gb)
      self.w = self.w - self.learning_rate * g / np.sqrt(G + eps)
      self.b = self.b - self.learning_rate * gb / np.sqrt(Gb + eps)


class RMSProp:
  def __init__(self, learning_rate, epoch_cnt, tao, p, n):
    self.learning_rate = learning_rate
    self.epoch_cnt = epoch_cnt
    self.tao = tao
    self.p = p
    self.n = n
    self.w = np.zeros(n)
    self.b = 0

  def predict(self, x):
    return sigmoid(np.dot(x, self.w) + self.b)

  def train(self, features, answers):
    G = np.zeros(self.n)
    Gb = 0
    eps = 1e-8
    for i in range(self.epoch_cnt):
      index = i % len(features)
      answer = answers[index]
      x = features[index]
      prediction = self.predict(x)
      g = (prediction - answer) * x + 2 * self.tao * self.w
      gb = (prediction - answer) + 2 * self.tao * self.b
      G = self.p * G + (1 - self.p) * np.square(g)
      Gb = self.p * Gb + (1 - self.p) * np.square(gb)
      self.w = self.w - self.learning_rate * g / np.sqrt(G + eps)
      self.b = self.b - self.learning_rate * gb / np.sqrt(Gb + eps)

class Adam:
  def __init__(self, learning_rate, epoch_cnt, tao, beta1, beta2, n):
    self.learning_rate = learning_rate
    self.epoch_cnt = epoch_cnt
    self.tao = tao
    self.beta1 = beta1
    self.beta2 = beta2
    self.n = n
    self.w = np.zeros(n)
    self.b = 0

  def predict(self, x):
    return sigmoid(np.dot(x, self.w) + self.b)

  def train(self, features, answers):
    m = np.zeros(self.n)
    v = np.zeros(self.n)
    mb = 0
    vb = 0
    eps = 1e-8
    for i in range(self.epoch_cnt):
      index = i % len(features)
      answer = answers[index]
      x = features[index]
      prediction = self.predict(x)
      g = (prediction - answer) * x + 2 * self.tao * self.w
      gb = (prediction - answer) + 2 * self.tao * self.b
      m = self.beta1 * m + (1 - self.beta1) * g
      mb = self.beta1 * mb + (1 - self.beta1) * gb
      v = self.beta2 * v + (1 - self.beta2) * np.square(g)
      vb = self.beta2 * vb + (1 - self.beta2) * np.square(gb)
      self.w = self.w - self.learning_rate * m / np.sqrt(v + eps)
      self.b = self.b - self.learning_rate * mb / np.sqrt(vb + eps)