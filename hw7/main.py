# -*- coding: utf-8 -*-
import numpy as np

import sgd
import bgd
import adagrad
import rmsprop
import adam
import sgd_momentum
import sgd_nesterov_momentum
from reader import read_irises
from quality import train_test

irises = read_irises('hw7/data/iris.csv')
np.random.shuffle(irises)
irises = train_test(irises)

print('##### Task 1 #####')
sgd.run(irises)

print('##### Task 2 #####')
bgd.run(irises)

print('##### Task 3 #####')
print('''Мне кажется тестовой выборки такого размера (15) недостаточно, чтобы сделать какой-то вывод о качестве SGD и BGD.
Интуитивно кажется, что BGD будет точнее, однако налицо длительность обучения: BGD обучается намного дольше, чем SGD,
однако результат не выглядит сильно отличающимся (если можно судить по такой маленькой выборке).

Также хочу заметить, что Iris-setosa отделяется лучше всех. Я посмотрел диаграмму рассеивания ирисов, и действительно,
первый класс лучше всех отделяется от второго и третьего по всем фичам. Чего не скажешь о втором,
потому что он находится между первым и третьим и явно неотделим линейно
''')

print('##### Task 4 #####')
sgd_momentum.run(irises)
sgd_nesterov_momentum.run(irises)
adagrad.run(irises)
rmsprop.run(irises)
adam.run(irises)