from csv import reader

DATA_DIR = 'data/'
TRAIN_DATA = DATA_DIR + 'train/'
TEST_DATA = DATA_DIR + 'test/'

TRAIN_INFO = DATA_DIR + 'train_info.csv'

def get_train_data(scale_dataset=100):
  '''
  opens TRAIN_INFO and retrieves the location and artist ID for each painting
  scales down dataset by <scale_dataset> if included
  '''
  with open(TRAIN_INFO, 'r') as csvfile:
    datafile = reader(csvfile, delimiter=',')
    headers = []
    data = []
    for row in datafile:
      if not len(headers):
        headers = row
      else:
        data.append(row[:2])
  return headers, data
  