from wrangledata import get_train_data
from PIL import Image

from keras.backend import variable as MakeTensor
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv3D, Dense, concatenate, Input, MaxPooling2D, Flatten
from sys import argv
import csv
import numpy as np
import os

# 60-64 kernel convolutions

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

    return model



def train_model(model, epochs, data_path, pairs_csv):
    # Create two inputs, each 4D tensors or batches of images
    # Also create a numpy array of y values for these pairs of images



    # (file a, file b, label)
    pairs = []

    with open(pairs_csv) as f:
        reader = csv.reader(f)
        # next(reader, None)  # Skipping the header
        pairs = [tuple(line) for line in reader]

    batch_size = 32

    train_split = int(len(pairs) * 0.6)

    pairs_train = pairs[:train_split]
    pairs_cv = pairs[train_split:int(len(pairs) * 0.8)]

    firstweights = str(model.get_weights())
    for i in range(10000):
        print('batch:', i)
        # TODO: remeber to handle when loading fails... currently filling with 0s... not ideal
        # perhaps this means not training on pairs with them...
        batch_a = np.zeros((batch_size, 100, 100, 3))
        batch_b = np.zeros((batch_size, 100, 100, 3))
        batch_y = np.zeros(batch_size)
        for j in range(batch_size):
            try:
                batch_a[j] = get_image(pairs_train[i * batch_size + j][0])

                batch_b[j] = get_image(pairs_train[i * batch_size + j][1])
            except:
                print('tgfrgtfgggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg')
            batch_y[j] = pairs_train[i * batch_size + j][2]

        model.train_on_batch([batch_a, batch_b], batch_y)
    print(str(model.get_weights()) == firstweights)

    # test image batches
    for i in range(10):
        print('batch:', i)
        # TODO: remeber to handle when loading fails... currently filling with 0s... not ideal
        # perhaps this means not training on pairs with them...
        test_batch_a = np.zeros((batch_size, 100, 100, 3))
        test_batch_b = np.zeros((batch_size, 100, 100, 3))
        test_batch_y = np.zeros(batch_size)
        for j in range(batch_size):
            try:
                test_batch_a[j] = get_image(pairs[i * batch_size + j][0])

                test_batch_b[j] = get_image(pairs[i * batch_size + j][1])
            except:
                print('tgfrgtfgggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg')
            test_batch_y[j] = pairs[i * batch_size + j][2]


        print('loss:', model.test_on_batch([test_batch_a, test_batch_b], test_batch_y))
        print(model.predict_on_batch([test_batch_a, test_batch_b]))
        print(test_batch_y)

def get_image(path):
    with Image.open(path) as image:
        image = np.array(image) / 255
        #print('image file a:', path)
        #print('single image shape a:', image.shape)
        if len(image.shape) == 2:
            image = np.dstack((image, image, image))

        return image.astype('float32')


def main():
    # TODO: don't make these sizes "magic numbers..." these are not random...
    model = create_model((100, 100, 3))
    train_model(model, 123, 23452, os.path.join('data', 'pairwise_train_info.csv'))

if __name__ == '__main__':
    main()





