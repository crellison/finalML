from wrangledata import get_train_data
from PIL import Image
from process_images import square_image

from keras.backend import variable as MakeTensor
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv3D, Dense, concatenate, Input, MaxPooling2D, Flatten
from sys import argv
import csv
import numpy as np
import os

# 60-64 kernel convolutions
IMAGE_DIR = 'data/train/'
IMAGE_DIM = 256
IMAGE_SHAPE = (256, 256, 3)

BATCH_SIZE = 32

def image_to_tensor(path):
    return MakeTensor(Image.open(path))

def create_model(input_shape):

    sub_net = create_sub_network(input_shape)

    input_a = Input(input_shape)
    input_b = Input(input_shape)

    # Build the feature sets outputed by each network
    features_a = sub_net(input_a)
    features_b = sub_net(input_b)

    # Concatenate the features into one vector for logistic regression
    merged_features = concatenate([features_a, features_b], axis=-1)

    # Logistic regression in place of a distance function.
    # Run logistic regression on the image's outputed features
    predictions = Dense(1, activation='sigmoid')(merged_features)

    # Make the model from the inputs and flow of the output
    model = Model(inputs=[input_a, input_b], outputs=predictions)

    # Compile the model. It should be ready to train
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def create_sub_network(input_shape):
    print('creating subnetwork...')
    model = Sequential()
    model.add(Conv2D(32, (9, 9), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (7, 7), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    print('subnetwork created')
    return model



def train_model(model, image_size, n_train_batches, data_path, pairs_csv):
    # Create two inputs, each 4D tensors or batches of images
    # Also create a numpy array of y values for these pairs of images

    pairs = []
    with open(pairs_csv) as f:
        reader = csv.reader(f)
        # next(reader, None)  # Skipping the header
        pairs = [tuple(line) for line in reader]

    train_split = int(len(pairs) * 0.6)

    train_pairs = pairs[:train_split]
    test_pairs = pairs[train_split:int(len(pairs) * 0.8)]

    firstweights = str(model.get_weights())

    for i in range(n_train_batches):

        train_batch_a, train_batch_b, train_batch_y = get_batch(train_pairs, i, BATCH_SIZE, image_size)

        model.train_on_batch([train_batch_a, train_batch_b], train_batch_y)

    print(str(model.get_weights()) == firstweights)

    # Test image batches
    for i in range(10):
        test_batch_a, test_batch_b, test_batch_y = get_batch(test_pairs, i, BATCH_SIZE, image_size)

        print('loss:', model.test_on_batch([test_batch_a, test_batch_b], test_batch_y))
        print(model.predict_on_batch([test_batch_a, test_batch_b]))
        print(test_batch_y)


def get_batch(pairs, batch_id, batch_size, image_size):
    batch_a = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
    batch_b = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
    batch_y = np.zeros(batch_size)
    print('getting batch:', batch_id)
    for j in range(batch_size):
        # TODO: Try the next image when the loading fails
        try:
            path1, path2, y = pairs[batch_id * batch_size + j]
            path1 = os.join.path(IMAGE_DIR, path1)
            path2 = os.join.path(IMAGE_DIR, path2)

            batch_a[j] = get_image(path1)
            batch_b[j] = get_image(path2)
        except:
            # TODO: Change this to a real exception
            print('FAILED TO GET IMAGE INTO BATCH', batch_id, ', Image #', j)
        batch_y[j] = pairs[batch_id * batch_size + j][2]
    
    return batch_a, batch_b, batch_y

def get_image(path):
    with Image.open(path) as image:
        print(image)
        image = square_image(image)
        image = np.array(image) / 255
        #print('image file a:', path)
        #print('single image shape a:', image.shape)
        if len(image.shape) == 2:
            image = np.dstack((image, image, image))

        return image.astype('float32')


def main():
    # TODO: Get size from images
    model = create_model(IMAGE_SHAPE)
    train_model(model, IMAGE_SHAPE, 1000, 'placeholder', os.path.join('data', 'pairwise_train_info.csv'))

if __name__ == '__main__':
    main()
    