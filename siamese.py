from PIL import Image
import process_images
from keras.backend import variable
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, Lambda, Dense, Input, MaxPooling2D, Flatten
import csv
import numpy as np
import os
import sys

import siamese_utils

IMAGE_DIM = 100

TRAIN_SPLIT = 0.6
CV_TEST_SPLIT = 0.8

BATCH_SIZE = 32

DATA_DIR = os.path.join('data', 'train')
OUTPUT_FILE = os.path.join('Models', 'painting_weights')


def create_model(input_shape, sub_net_choice):
    """
    Returns a siamese model with a subnet work chosen with sub_net_choice:
        'large': cnn with 4 convolution layers
    """
    twin_net = {'large': create_large_sub_net}[sub_net_choice](input_shape)

    input_a = Input(input_shape)
    input_b = Input(input_shape)

    # Build the feature sets outputted by each network
    features_a = twin_net(input_a)
    features_b = twin_net(input_b)

    # Compute the distance between the processed features for each image
    distance = Lambda(siamese_utils.euclidean_distance, output_shape=siamese_utils.eucl_dist_output_shape)([features_a, features_b])

    # Create model from the inputs and flow of the output
    model = Model(inputs=[input_a, input_b], outputs=distance)

    # Compile the model
    print('Compiling model')
    model.compile(optimizer='rmsprop', loss=siamese_utils.contrastive_loss, metrics=['accuracy'])

    return model


def create_large_sub_net(input_shape):
    print('creating subnetwork...')
    model = Sequential()

    model.add(Conv2D(32, (9, 9), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (7, 7), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    # model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def train_model(model, image_size, n_train_batches, pairs_csv):

    # Open the pre-made file of pairs
    with open(pairs_csv) as pairs_file:
        reader = csv.reader(pairs_file)
        pairs = [tuple(line) for line in reader]

    # Split the pairs file into the three sets, Train, CV, and Test
    train_split = int(len(pairs) * TRAIN_SPLIT)
    cv_test_split = int(len(pairs) * CV_TEST_SPLIT)

    train_pairs = pairs[:train_split]
    cv_pairs = pairs[train_split:cv_test_split]
    test_pairs = pairs[cv_test_split:]

    # Train the model
    for batch in range(n_train_batches):
        print('Training batch', batch)
        train_batch_a, train_batch_b, train_batch_y = get_batch(train_pairs, (batch * BATCH_SIZE) % len(pairs), image_size)

        model.train_on_batch([train_batch_a, train_batch_b], train_batch_y)

    model.save(OUTPUT_FILE)

    # Test image batches
    all_predictions = np.ndarray((1, 1))
    all_y = np.ndarray((1, 1))

    for batch in range(10):
        print('CV batch', batch)
        test_batch_a, test_batch_b, test_batch_y = get_batch(cv_pairs, (batch * BATCH_SIZE) % len(pairs), image_size)
        predictions_batch = model.predict_on_batch([test_batch_a, test_batch_b])

        all_predictions = np.vstack((all_predictions, np.round(predictions_batch)))
        all_y = np.vstack((all_y, test_batch_y.reshape(BATCH_SIZE, 1)))


    siamese_utils.eval_siamese(all_y[1:], all_predictions[1:])


def get_batch(pairs, batch_id, image_size):
    batch_a = np.zeros((BATCH_SIZE, image_size[0], image_size[1], image_size[2]), dtype="float32")
    batch_b = np.zeros((BATCH_SIZE, image_size[0], image_size[1], image_size[2]), dtype="float32")
    batch_y = np.zeros(BATCH_SIZE, dtype="float32")

    for j in range(BATCH_SIZE):
        batch_a[j] = get_image(pairs[batch_id * BATCH_SIZE + j][0])
        batch_b[j] = get_image(pairs[batch_id * BATCH_SIZE + j][1])

        # TODO: Try the next image when the loading fails
        try:
            batch_b[j] = get_image(pairs[batch_id * BATCH_SIZE + j][1])
            batch_a[j] = get_image(pairs[batch_id * BATCH_SIZE + j][0])
        except:
            # TODO: Change this to a real exception
            print('FAILED TO GET IMAGE INTO BATCH', batch_id, ', Image #', j)
        batch_y[j] = pairs[batch_id * BATCH_SIZE + j][2]

    return batch_a, batch_b, batch_y

def get_image(path):
    with Image.open(path) as image:
        image = process_images.square_image(image)
        image = image.resize((IMAGE_DIM, IMAGE_DIM))
        image = np.array(image) / 255

        # Convert gray-scale images to RGB
        if len(image.shape) == 2:
            image = np.dstack((image, image, image))

        return image.astype('float32')

def image_to_tensor(path):
    return variable(Image.open(path))


def load_trained_model(pretrained_model_file):
    # because it is a custom loss function we need to provide that through "custom_objects"
    return load_model(pretrained_model_file, custom_objects={'contrastive_loss': siamese_utils.contrastive_loss})

def competition_test_model(model, pairs_csv):
    # Open the pre-made file of pairs
    with open(pairs_csv) as pairs_file:
        reader = csv.reader(pairs_file)
        next(reader, None) # Skips the header.
        pairs = [tuple(line) for line in reader]

        print(len(pairs))

    # there are 21916047 pairs in the test set

def main():
    # This condition will change depending on what params, we'd like to play with:
    if len(sys.argv) < 2:
        model = create_model((IMAGE_DIM, IMAGE_DIM, 3), 'large')
        print(model.summary())
        train_model(model, (IMAGE_DIM, IMAGE_DIM, 3), 500, os.path.join('data', 'pairwise_train_info.csv'))

    # Pre-trained model provided:
    else:
        model = load_trained_model(sys.argv[1])

        # competition_test_model(3, os.path.join('data', 'submission_info.csv'))
#         assumes: submission info in data
#         TODO: run on testing set






if __name__ == '__main__':
    main()






# %USERPROFILE%/.keras/keras.json
