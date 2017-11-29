from csv import reader
from random import shuffle, choice, choices, sample

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
      filename = TRAIN_DATA + row[0] 
      painterID = row[1]
      data.append((filename, painterID))

  datalen = int(len(data) * scale_dataset)
  data = data[:datalen]

  return headers, data
  
def create_pairs():
  m = 0
  data = dict()
  artists = set()
  with open(SORTED_INFO, 'r') as csvfile:
    datafile = reader(csvfile, delimiter=',')
    for row in datafile:
      m += 1
      filepath, artistID = row
      artists.add(artistID)
      if artistID not in data:
        data[artistID] = [filepath]
      else:
        data[artistID].append(filepath)

  artists = list(artists)
  artist_weights = [len(data[ID]) for ID in data]
  print('%i differnt artists with %i total paintings' % (len(artists), m))
  pairs = []
  for artistID in artists:
    num_paintings = len(data[artistID])
    num_pairs = min(MAX_ARTIST_PAIRS, num_paintings // 2)

    # collect num_pairs pairs of paintings from the same artist
    # tuples of form (path, path, y) with y=1 if same artist else y=0
    cur_paintings = data[artistID]
    same_pairs = [(cur_paintings[i], cur_paintings[num_paintings-1-i], '1') for i in range(num_pairs)]
    pairs.extend(same_pairs)

    # collect 1 training example per painting with different artists
    diff_pairs = []
    for painting in data[artistID]:
      rand_artist = choices(artists, weights=artist_weights)[0]
      while rand_artist == artistID:
        rand_artist = choices(artists, weights=artist_weights)[0]
      rand_painting = choice(data[rand_artist])
      diff_pairs.append((painting, rand_painting, '0'))

    pairs.extend(diff_pairs)

  return pairs

def main():
  pairs = create_pairs()
  m = len(pairs)
  pos = sum(int(pair[2]) for pair in pairs)
  print('%i training points with %i positive examples' % (m, pos))
  with open(PAIR_TRAIN_SET, 'w+') as writer:
    for pair in pairs:
      writer.write(','.join(pair)+'\n')

if __name__ == '__main__':
  main()
  