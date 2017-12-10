import process_images
from keras.backend import variable
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, Lambda, Dense, Input, MaxPooling2D, Flatten
import csv
import os
import sys

import siamese_utils
from siamese_utils import *

IMAGE_DIM = 200

TRAIN_SPLIT = 0.8
CV_TEST_SPLIT = 0.9

BATCH_SIZE = 8

DATA_DIR = os.path.join('data', 'train')
OUTPUT_FILE = os.path.join('Models', 'painting_model.h5')


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
    model = Sequential()

    model.add(Conv2D(32, (9, 9), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (7, 7), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, image_size, epochs, pairs_csv):

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
    batches_per_epoch = len(train_pairs) // BATCH_SIZE
    print('Total batches to train', batches_per_epoch * epochs)
    print('train batches', len(train_pairs) // BATCH_SIZE)
    fractional_epoch = epochs - np.floor(epochs)
    epochs = int(np.floor(epochs))

    # Train full epochs
    for epoch in range(epochs):
        print('Training epoch', epoch)
        for batch in range(batches_per_epoch):
            print('Training batch', batches_per_epoch * epoch + batch)
            train_batch_a, train_batch_b, train_batch_y = get_batch(train_pairs, (batch * BATCH_SIZE), image_size)
            model.train_on_batch([train_batch_a, train_batch_b], train_batch_y)

    # Train remaining batches
    print('fractional epoch', fractional_epoch)
    print('batches to train', int(batches_per_epoch * fractional_epoch))
    for batch in range(int(batches_per_epoch * fractional_epoch)):
        print('Training batch', batches_per_epoch * epochs + batch)
        train_batch_a, train_batch_b, train_batch_y = get_batch(train_pairs, (batch * BATCH_SIZE), image_size)
        model.train_on_batch([train_batch_a, train_batch_b], train_batch_y)

    # Save the model for
    print('Training finished. Saving model.')
    model.save(OUTPUT_FILE)


    # CV image batch collectors, first element ignored later
    cv_predictions = np.ndarray((1, 1))
    cv_y = np.ndarray((1, 1))

    for batch in range(100):
        print('CV batch', batch)
        cv_batch_a, cv_batch_b, cv_batch_y = get_batch(cv_pairs, (batch * BATCH_SIZE), image_size)
        predictions_batch = model.predict_on_batch([cv_batch_a, cv_batch_b])

        cv_predictions = np.vstack((cv_predictions, np.round(predictions_batch)))
        cv_y = np.vstack((cv_y, cv_batch_y.reshape(BATCH_SIZE, 1)))


    siamese_utils.eval_siamese(cv_y[1:], cv_predictions[1:])


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

# TODO:
def generate_predictions(model, pairs_csv):
    # Open the pre-made file of pairs
    with open(pairs_csv) as pairs_file:
        reader = csv.reader(pairs_file)
        next(reader, None) # Skips the header.
        pairs = [tuple(line) for line in reader]

        print(len(pairs))

    output = open('comp_output.txt', 'w+')

    output.write('index, sameArtist\n')


    for batch in range(100):
        batch_index, batch_a, batch_b = get_competition_batch(pairs, batch)
        predictions_batch = model.predict_on_batch([batch_a, batch_b])
        for prediction in range(len(predictions_batch)):
            out = '%d' % batch_index[prediction] + ',' + str(predictions_batch[prediction][0]) + '\n'
            output.write(out)


    # leftovers...

    output.close()

    # there are 21916047 pairs in the test set



def get_competition_batch(pairs, batch_id):
    batch_a = np.zeros((BATCH_SIZE, IMAGE_DIM, IMAGE_DIM, 3), dtype="float32")
    batch_b = np.zeros((BATCH_SIZE, IMAGE_DIM, IMAGE_DIM, 3), dtype="float32")
    batch_index = np.zeros(BATCH_SIZE, dtype="float32")

    for j in range(BATCH_SIZE):
        #
        # Configure this to fit your file structure...
        #
        batch_a[j] = get_image(os.path.join('data', 'test', pairs[batch_id * BATCH_SIZE + j][1]))
        batch_b[j] = get_image(os.path.join('data', 'test', pairs[batch_id * BATCH_SIZE + j][2]))

        # error will crash this...

        batch_index[j] = pairs[batch_id * BATCH_SIZE + j][0]

    return batch_index, batch_a, batch_b

def main():
    # This condition will change depending on what params, we'd like to play with:
    if len(sys.argv) < 2:
        model = create_model((IMAGE_DIM, IMAGE_DIM, 3), 'large')
        print(model.summary())
        train_model(model, (IMAGE_DIM, IMAGE_DIM, 3), 0.0005, os.path.join('data', 'pairwise_train_info.csv'))

    # Pre-trained model provided:
    else:
        # If the passed in file is a whole model, not just weights:
        model = load_trained_model(os.path.join('models','painting_model.h5'))
        generate_predictions(model, os.path.join('data', 'submission_info.csv'))


if __name__ == '__main__':
    main()






# %USERPROFILE%/.keras/keras.json
