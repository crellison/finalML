from csv import reader
from random import shuffle

DATA_DIR = 'data/'
TRAIN_DATA = DATA_DIR + 'train/'
TEST_DATA = DATA_DIR + 'test/'

TRAIN_INFO = DATA_DIR + 'train_info.csv'

def get_train_data(scale_dataset=1, randomize=True):
  '''
  opens TRAIN_INFO and retrieves the location and artist ID for each painting
  scales down dataset by <scale_dataset> if included
  '''
  assert 0 < scale_dataset <= 1

  with open(TRAIN_INFO, 'r') as csvfile:
    datafile = reader(csvfile, delimiter=',')
    headers = []
    data = []

    totaltraindata = sum(1 for row in datafile) - 1
    datalen = int(totaltraindata * scale_dataset)

    print('retrieving %i of %i painting from training set' % (datalen, totaltraindata))

    linenum = 0
    for row in datafile:
      if not len(headers):
        headers = row
      else:
        data.append(row[:2])
        linenum += 1

      if not randomize and linenum >= datalen:
        break

  if randomize:
    shuffle(data)
    data = data[:datalen]

  return headers, data