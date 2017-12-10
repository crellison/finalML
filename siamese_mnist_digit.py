from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, Dense, concatenate, Input, MaxPooling2D, Flatten, Dropout, Lambda
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import to_categorical
from keras.optimizers import RMSprop
import sys
import numpy as np
from siamese_utils import *
from cnn_models import *

MNIST_SHAPE = (28, 28, 1)


def siameseCNN(sub_net, input_shape):
    """
    builds a siamese CNN with given subnet and input_shape
    """

    input_a = Input(input_shape)
    input_b = Input(input_shape)

    features_a = sub_net(input_a)
    features_b = sub_net(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([features_a, features_b])

    model = Model(inputs=[input_a, input_b], outputs=distance)

    rms = RMSprop()
    model.compile(optimizer=rms, loss=contrastive_loss)  # 'binary_crossentropy'

    return model


def gen_siamese_pairs(pairs, data):
    """
    generates list of pair data from pair index list
    """
    x_train_a = []
    x_train_b = []

    for item in pairs:
        x_train_a.append(data[item[0]])
        x_train_b.append(data[item[1]])

    x_train_a = np.array(x_train_a).astype('float32')
    x_train_b = np.array(x_train_b).astype('float32')

    rows, cols, depth = MNIST_SHAPE
    x_train_a = x_train_a.reshape(x_train_a.shape[0], rows, cols, depth)
    x_train_b = x_train_b.reshape(x_train_b.shape[0], rows, cols, depth)

    x_train_a = x_train_a.astype('float32')
    x_train_b = x_train_b.astype('float32')

    x_train_a /= 255
    x_train_b /= 255

    return x_train_a, x_train_b


def trainSiamese(with_GPU=False, pre_trained_file=''):
    """
    trains a siamese CNN with data from MNIST, or loads a pretrained model
    """

    # Loading Image Data:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Siamese net training:
    if pre_trained_file == '':
        sub_net = five_convolutions(MNIST_SHAPE) if with_GPU else three_convolutions(MNIST_SHAPE)

        print('\nSUBNET ARCHITECTURE')
        print(sub_net.summary())

        model = siameseCNN(sub_net, MNIST_SHAPE)
        print('\nSIAMESE ARCHITECTURE')
        print(model.summary())
        print('\n')

        x_train = np.array(x_train)
        x_train_hash = make_hash(y_train)
        train_labels, train_pairs = create_pairs(x_train_hash)
        x_train_a, x_train_b = gen_siamese_pairs(train_pairs, x_train)

        model.fit([x_train_a, x_train_b], train_labels,
                  verbose=1,
                  epochs=5, batch_size=32,
                  validation_split=0.2)

        model_file = outfile('mnist_digit_siamese')
        print('Saving trained model to: ' + model_file)
        model.save(model_file)

    # Skip Training If a Pre-trained Model is Provided:
    else:
        model = load_model(pre_trained_file, custom_objects={'contrastive_loss': contrastive_loss})

        print('\nSIAMESE ARCHITECTURE')
        print(model.summary())
        print('\n')
        print("Using the provided, pre-trained model")

    # Testing the Model:
    x_test = np.array(x_test)
    x_test_hash = make_hash(y_test)
    test_labels, test_pairs = create_pairs(x_test_hash)

    x_test_a, x_test_b = gen_siamese_pairs(test_pairs, x_test)

    test_pred = model.predict([x_test_a, x_test_b], batch_size=128)
    eval_siamese(test_pred, test_labels)


def trainCNN():
    """
    trains a CNN with MNIST data
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # y_train = np.array(y_train).astype('float32')
    # y_test = np.array(y_test).astype('float32')

    img_rows, img_cols = 28, 28
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # x is image, y is label
    # model = fashion_network()
    model = three_convolutions(MNIST_SHAPE)
    print(model.summary())
    model.fit(x_train, y_train, verbose=1,
              epochs=10, batch_size=32,
              validation_split=0.2)
    # validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, batch_size=128)
    print('loss: %f \t accuracy: %f' % tuple(score))

    model.save(outfile('mnist_digit_CNN'))


def main():
    # trainCNN()
    pre_trained_file = '' if len(sys.argv) < 2 else sys.argv[1]
    trainSiamese(pre_trained_file=pre_trained_file)


if __name__ == '__main__':
    main()
