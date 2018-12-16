import commonutil
import scipy


def process_line(line, label):
  attrs = line.rstrip().split(' ')
  result = label[:-1]
  for i, v in enumerate(attrs):
    result = '{} {}:{}'.format(result, i + 1, v)
  
  return result + '\n'


labels = []
train_data = []
with open('./data/gisette/gisette_train.labels', 'r') as input:
  for label in input:
    labels.append(label)

with open('./data/gisette/gisette_train.data', 'r') as input:
  for i, line in enumerate(input):
    train_data.append(process_line(line, labels[i]))

with open('./data/gisette/gisette.train.data', 'w') as output:
  for line in train_data:
    output.write(line)

TRAIN_SIZE = len(train_data)
print('Train length: {}'.format(TRAIN_SIZE))

scale_data = train_data
with open('./data/gisette/gisette_test.data', 'r') as input:
  for i, line in enumerate(input):
    scale_data.append(process_line(line, '-1\n'))

with open('./data/gisette/gisette.scale.data', 'w') as output:
  for line in scale_data:
    output.write(line)

SCALE_SIZE = len(scale_data)
print('Scale length: {}'.format(SCALE_SIZE))

labels = []
test_data = []
with open('./data/gisette/gisette_valid.labels', 'r') as input:
  for label in input:
    labels.append(label)

with open('./data/gisette/gisette_valid.data', 'r') as input:
  for i, line in enumerate(input):
    test_data.append(process_line(line, labels[i]))

with open('./data/gisette/gisette.test.data', 'w') as output:
  for line in test_data:
    output.write(line)

TEST_SIZE = len(test_data)
print('Test length: {}'.format(TEST_SIZE))

