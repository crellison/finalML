from csv import reader
from random import shuffle, choice, sample

DATA_DIR = 'data/'
TRAIN_DATA = DATA_DIR + 'train/'
TEST_DATA = DATA_DIR + 'test/'

TRAIN_INFO = DATA_DIR + 'train_info.csv'
SORTED_INFO = DATA_DIR + 'train_info_sorted.csv'

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
  print('%i differnt artists with %i total paintings' % (len(artists), m))
  pairs = []
  for artistID in artists:
    num_paintings = len(data[artistID])
    num_pairs = min(MAX_ARTIST_PAIRS, num_paintings // 2)

    # collect num_pairs pairs of paintings from the same artist
    # tuples of form (path, path, y) with y=1 if same artist else y=0
    cur_paintings = sample(data[artistID], num_paintings)
    same_pairs = [(cur_paintings[i], cur_paintings[num_paintings-1-i], 1) for i in range(num_pairs)]
    pairs.extend(same_pairs)

    # collect num_pairs training examples with different artists
    diff_pairs = []
    for i in range(num_pairs):
      cur_painting = choice(cur_paintings)
      rand_artist = choice(artists)
      while rand_artist == artistID:
        rand_artist = choice(artists)
      rand_painting = choice(data[rand_artist])
      diff_pairs.append((cur_painting, rand_painting, 0))

    pairs.extend(diff_pairs)

  return pairs

def main():
  pairs = create_pairs()
  print(pairs[1])
  print(pairs[100])
  print(pairs[300])
  print(pairs[1000])
  print(pairs[4000])

if __name__ == '__main__':
  main()