import commonutil
import scipy

train = []
test = []
TRAIN_SIZE = 3450
TEST_SIZE = 1151


def process_line(line):
  attrs = line.split(',')
  result = attrs.pop()[:-1]
  for i, v in enumerate(attrs):
    result = '{} {}:{}'.format(result, i + 1, v)
  
  return result + '\n'

with open('./data/spambase/spambase.data.shuffled', 'r') as input:
  for _ in range(TRAIN_SIZE):
    train.append(process_line(input.readline()))
  for _ in range(TEST_SIZE):
    test.append(process_line(input.readline()))


with open('./data/spambase/spambase.train', 'w') as train_output:
  for line in train:
    train_output.write(line)

with open('./data/spambase/spambase.test', 'w') as test_output:
  for line in test:
    test_output.write(line)
