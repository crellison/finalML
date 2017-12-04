from PIL import Image

import numpy as np
import csv
import os
import random

# fname = "/Users/HenrySwaffield/Documents/Middlebury Senior Year/fall/CS451/FinalProject/data/small/train/47.jpg"
# image = np.array(Image.open(fname))

def read_image(filename):
    image = np.array(Image.open(filename))
    return image

def normalize_image(image):
    # TODO: Consider other normalization techniques
    return image / 255.0

def create_pairs(metadata):
    m = len(metadata)
    pairs = []
    current_artist = ""
    artist_begin_index = 0
    artist_end_index = 1
    i = 0
    while i < m:

        new_artist = metadata[i][1]
        if new_artist != current_artist:
            current_artist = new_artist
            artist_begin_index = i

            # Find the end of the current artist's section
            for a in range(artist_begin_index, m):
                if metadata[a][1] != current_artist:
                    artist_end_index = a - 1
                    # i = a
                    break

            print('end index: %i' % artist_end_index)

            # TODO: Add the possibility to choose different numbers of matching and not matching artists

            n_same_artist = min(10000, artist_end_index - artist_begin_index)
            # TODO: rather than just loop and make random pairs, create all the combinations...
            for num_pairs in range(n_same_artist):
                # TODO: Make sure that making a twin pair of images doesn't cause problems
                same_artist_pair = (random_painting(metadata, artist_begin_index, artist_end_index), random_painting(metadata, artist_begin_index, artist_end_index))
                pairs.append(same_artist_pair)

                # Add a pair of paintings by different artists
                choose_from_before = random.random() > float(artist_begin_index) / float(m)

                # if choose_from_before:
                #     other_painting_index = random.randint(0, artist_begin_index)
                # else:
                #     other_painting_index = random.randint(artist_end_index, m-1)

                other_painting_index = artist_begin_index
                while other_painting_index >= artist_begin_index and other_painting_index <= artist_end_index:
                    other_painting_index = random.randint(0, m-1)

                painting_one = metadata[other_painting_index]
                pairs.append((painting_one, random_painting(metadata, artist_begin_index, artist_end_index)))
        i += 1

    for r in pairs:
        print(r)

def random_painting(metadata, min_index, max_index):
    '''
    retrieves a random painting from metadata between min_index and max_index
    :param metadata: list of tuples
    :param min_index: int
    :param max_index: int
    :return:
    '''
    assert min_index < max_index

    return metadata[random.randint(min_index, max_index)]

def read_metadata():
    meta_data = []
    path = os.path.join("data", "train_info_sorted.csv")

    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)  # Skipping the header
        meta_data = [tuple(line) for line in reader]

    return meta_data

if __name__ == '__main__':
    create_pairs(read_metadata())
