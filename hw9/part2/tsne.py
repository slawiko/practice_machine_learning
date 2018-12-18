import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import fetch_mldata
from sklearn.manifold import TSNE


def plot(title, x, y, colors):
  plt.title(title)
  plt.figure(figsize=(25, 15))

  plt.scatter(x, y, c=colors, alpha=0.6, cmap=cm.get_cmap('jet', 10))
  plt.colorbar(ticks=range(10))
  plt.clim(-0.5, 9.5)

  plt.savefig('./hw9/part2/plots/' + title)
  plt.gcf().clear()

def tSNE(features, labels):
  perplexities = [10, 30, 50]
  iterations = [250, 500, 1000, 3000]
  learning_rates = [500]
  for perplexity in perplexities:
    for iteration_cnt in iterations:
      for learning_rate in learning_rates:
        print('p = {}, iterations = {} starting...'.format(perplexity, iteration_cnt))
        title = 'p={}_iterations={}'.format(perplexity, iteration_cnt, learning_rate)
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=iteration_cnt, learning_rate=learning_rate)
        tsne_results = tsne.fit_transform(features)
        plot(title, tsne_results[:, 0], tsne_results[:, 1], labels)
        print('p = {}, iterations = {} finished.'.format(perplexity, iteration_cnt))

def prepare_data(size):
  mnist = fetch_mldata('MNIST original', data_home='./data/mnist')
  data_x = mnist['data']
  data_y = mnist['target']

  indices = np.random.permutation(len(data_x))
  data_x = data_x[indices]
  data_y = data_y[indices]

  return data_x[:size], data_y[:size]

PART_SIZE = 10000
x, y = prepare_data(PART_SIZE)
tSNE(x, y)
