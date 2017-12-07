from csv import reader
from random import shuffle, choice, choices, sample
from sys import stdout

DATA_DIR = 'data/'
TRAIN_DATA = DATA_DIR + 'train/'
TEST_DATA = DATA_DIR + 'test/'

TRAIN_INFO = DATA_DIR + 'train_info.csv'
SORTED_INFO = DATA_DIR + 'train_info_sorted.csv'
PAIR_TRAIN_SET = DATA_DIR + 'pairwise_train_info.csv'

MAX_ARTIST_PAIRS = 20

def get_train_data(scale_dataset=1.0):
  '''
  opens TRAIN_INFO and retrieves the location and artist ID for each painting
  scales down dataset by <scale_dataset> if included

  scale_dataset <float> used to resize the datasdet for debugging/testing purposes
  '''
  assert 0.0 < scale_dataset <= 1.0, 'scale_dataset must be between 0 and 1'

  data = []
  headers = []
  with open(TRAIN_INFO, 'r') as csvfile:
    datafile = reader(csvfile, delimiter=',')

    linenum = 0
    headers = next(datafile, None)
    for row in datafile:
      filename = row[0] 
      painterID = row[1]
      data.append((filename, painterID))

  datalen = int(len(data) * scale_dataset)
  data = data[:datalen]

  return headers, data
  
def create_pairs():
  m = 0
  data = dict()
  with open(SORTED_INFO, 'r') as csvfile:
    datafile = reader(csvfile, delimiter=',')
    for row in datafile:
      m += 1
      filepath, artistID = row
      if artistID not in data:
        data[artistID] = [filepath]
      else:
        data[artistID].append(filepath)

  artist_weights = [len(data[ID]) for ID in data]
  artists = list(data)
  n = len(artists)
  print('%i differnt artists with %i total paintings' % (n, m))
  pairs = []
  prog = 0
  proglen = 78
  for artistID in artists:
    prog += 1
    progress = int(prog/n * proglen)
    bar = '=' * progress
    empty = ' ' * (proglen-progress)
    stdout.write('\r[%s%s]' % (bar, empty))

    num_pairs = sum(x for x in range(len(data[artistID])))

    # collect all possible pairs of paintings from the same artist
    # tuples of form (path, path, y) with y=1 if same artist else y=0
    painting_pairs = all_pairs(data[artistID])
    same_pairs = [(pair[0], pair[1], '1') for pair in painting_pairs]
    pairs.extend(same_pairs)

    # collect factorial(num_pairs) training example per painting with different artists
    # should have same number diff artist pairs as same artist pairs
    diff_pairs = []
    
    for i in range(num_pairs):
      cur_painting = choice(data[artistID])

      rand_artist = choices(artists, weights=artist_weights)[0]
      while rand_artist == artistID:
        rand_artist = choices(artists, weights=artist_weights)[0]

      rand_painting = choice(data[rand_artist])
      diff_pairs.append((cur_painting, rand_painting, '0'))

    pairs.extend(diff_pairs)

  stdout.flush()
  print()
  return pairs

def all_pairs(item_list):
  pairs = []
  for i in range(len(item_list)-1):
    pairs.extend([(item_list[i], item) for item in item_list[i:]])
  return pairs

def main():
  pairs = create_pairs()
  print('pairs generated, shuffling pairs')
  shuffle(pairs)
  m = len(pairs)
  pos = sum(int(pair[2]) for pair in pairs)
  print('%i training points with %1.4f percent positive examples' % (m, pos/m*100))
  with open(PAIR_TRAIN_SET, 'w+') as writer:
    for pair in pairs:
      writer.write(','.join(pair)+'\n')

if __name__ == '__main__':
  main()
  