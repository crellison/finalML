from csv import reader
from random import shuffle

DATA_DIR = 'data/'
TRAIN_DATA = DATA_DIR + 'train/'
TEST_DATA = DATA_DIR + 'test/'

TRAIN_INFO = DATA_DIR + 'train_info.csv'

def get_train_data(scale_dataset=1.0, randomize=True):
  '''
  opens TRAIN_INFO and retrieves the location and artist ID for each painting
  scales down dataset by <scale_dataset> if included

  scale_dataset <float> used to resize the datasdet for debugging/testing purposes
  randomize <bool> determined whether or not the data is shuffled before it is delivered
  '''
  assert 0.0 < scale_dataset <= 1.0
  assert isinstance(randomize, bool)

  with open(TRAIN_INFO, 'r') as csvfile:
    datafile = reader(csvfile, delimiter=',')
    headers = []
    data = []

    totaltraindata = sum(1 for row in datafile) - 1
    datalen = int(totaltraindata * scale_dataset)

    print('retrieving %i of %i painting from training set' % (datalen, totaltraindata))

    linenum = 0
    for row in datafile:
      # grab the data-fields from the first row
      if not len(headers):
        headers = row
      # all subsequent rows are data
      else:
        filename = TRAIN_DATA + row[0] 
        painterID = row[1]
        data.append((filename, painterID))
        linenum += 1

      # randomization happens with all datapoints, so only cut out early
      # if randomization is not requested
      if not randomize and linenum >= datalen:
        break

  # shuffle and shorted the data
  if randomize:
    shuffle(data)
    data = data[:datalen]

  return headers, data