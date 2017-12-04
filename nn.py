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

    for i in range(100):
        batch_a = np.array((batch_size, 100, 100, 3))
        batch_b = np.array((batch_size, 100, 100, 3))
        batch_y = np.array(batch_size)
        for j in range(batch_size):
            with Image.open(pairs[i * batch_size + j][0]) as image_a:
                single_image = np.array(image_a) / 255
                print(single_image)
                batch_a[i] = single_image
            with Image.open(pairs[i * batch_size + j][1]) as image_b:
                batch_b[i] = np.array(image_b) / 255
            batch_y[j] = pairs[i * batch_size + j][2]

        model.train_on_batch([batch_a, batch_b], batch_y)




def main():
    image_a = argv[1]
    image_b = argv[2]
    model = create_model((100, 100, 3))
    train_model(model, 123, 23452, os.path.join('data', 'pairwise_train_info.csv'))

if __name__ == '__main__':
    main()