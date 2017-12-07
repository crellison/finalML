from keras.datasets import fashion_mnist
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, concatenate, Input, MaxPooling2D, Flatten, Dropout, Lambda
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import to_categorical
from keras.optimizers import RMSprop

from sklearn.metrics import accuracy_score, confusion_matrix

from math import floor
from time import time
from random import sample, randint, choice
import numpy as np

from nn import euclidean_distance, eucl_dist_output_shape, contrastive_loss

from PIL import Image as pImage

MNIST_SHAPE = (28, 28, 1)

def show_img(img):
  img_ = img * 255
  img_ = pImage.fromarray(img_, mode='L')
  img_.show(command='fim')

def make_hash(y_data):
  label_map = {x:list() for x in range(10)}
  for i in range(len(y_data)):
    label = y_data[i]
    label_map[label].append(i)
  return label_map

def create_pairs(label_map):
  smallest_group = min(len(label_map[label]) for label in range(10)) - 1
  pairs = []
  labels = []
  for label in range(10):
    for pair_num in range(smallest_group):
      a,b = sample(label_map[label], 2)

      other_label = randint(0,9)

      if other_label == label:
        other_label = (other_label + 1) % 10

      c = choice(label_map[label])
      d = choice(label_map[other_label])

      pairs.extend([(a,b), (c,d)])
      labels.extend([1,0])

  return labels, pairs

def siameseCNN(sub_net):

  input_a = Input(MNIST_SHAPE)
  input_b = Input(MNIST_SHAPE)

  features_a = sub_net(input_a)
  features_b = sub_net(input_b)

  # merged_features = concatenate([features_a, features_b], axis=-1)

  distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([features_a, features_b])

  model = Model(inputs=[input_a, input_b], outputs=distance)

  rms = RMSprop()
  model.compile(optimizer=rms, loss=contrastive_loss) #'binary_crossentropy'

  return model

def small_fashion_network():
  '''
  smaller CNN model for use w/o GPU
  '''
  model = Sequential()
  # input 28x28
  model.add(Conv2D(16, (5, 5), padding='same', activation='relu', input_shape=MNIST_SHAPE))
  model.add(MaxPooling2D())
  # input 14 x 14
  model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
  model.add(MaxPooling2D())
  model.add(MaxPooling2D())
  # input 7 x 7
  model.add(Conv2D(32, (3, 3), activation='relu'))
  # input 5 x 5
  model.add(Flatten())
  # input 800
  model.add(Dense(64, activation='relu'))

  model.add(Dense(10, activation='sigmoid'))
  model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  return model

def fashion_network():
  # input 28x28
  model = Sequential()
  model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=MNIST_SHAPE))
  model.add(Dropout(0.1))
  model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
  model.add(Dropout(0.1))
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D())
  model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D())
  model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(10, activation='sigmoid'))
  model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  return model

def make_data_from_pairs(pairs, data):
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

def outfile():
  signature = floor(time())
  return 'fashion_model_%i.h5' % signature

def trainSiamese(withGPU=False):
  sub_net = fashion_network() if withGPU else small_fashion_network()

  print('SUBNET ARCHITECTURE')
  print(sub_net.summary())
  print('\n')

  model = siameseCNN(sub_net)
  print('SIAMESE ARCHITECTURE')
  print(model.summary())
  print('\n')
  
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

  x_train = np.array(x_train)
  x_test = np.array(x_test)

  x_train_hash = make_hash(y_train)
  x_test_hash = make_hash(y_test)

  train_labels, train_pairs = create_pairs(x_train_hash)
  test_labels, test_pairs = create_pairs(x_test_hash)

  # train_labels = to_categorical(train_labels)
  # test_labels = to_categorical(test_labels)

  x_train_a, x_train_b = make_data_from_pairs(train_pairs, x_train)

  model.fit([x_train_a, x_train_b], train_labels,
            verbose=1,
            epochs=10, batch_size=32,
            validation_split=0.2)

  x_test_a, x_test_b = make_data_from_pairs(test_pairs, x_test)

  y_pred = model.predict([x_test_a, x_test_b], batch_size=128)
  y_pred = y_pred < 0.5
  y_truth = np.array(test_labels)
  # y_pred = np.argmax(y_pred, axis=0)
  # y_truth = np.argmax(test_labels, axis=0)

  print()
  print("accuracy is: " + str(accuracy_score(y_pred, y_truth)))
  print("confusion matrix:")
  print(confusion_matrix(y_pred, y_truth))

  model.save_weights(outfile())

def trainCNN():
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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
  model = small_fashion_network()
  print(model.summary())
  model.fit(x_train, y_train, verbose=1,
            epochs=10, batch_size=32,
            validation_split=0.2)
            # validation_data=(x_test, y_test))

  score = model.evaluate(x_test, y_test, batch_size=128)
  print('loss: %f \t accuracy: %f' % tuple(score))

def main():
  # trainCNN()
  trainSiamese()

if __name__ == '__main__':
  main()