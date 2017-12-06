from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, concatenate, Input, MaxPooling2D, Flatten
from keras.utils import to_categorical

from random import sample, randint, choice
import numpy as np


MNIST_SHAPE = (28, 28, 1)

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

def fashion_network():
  # input 28x28
  model = Sequential()
  model.add(Conv2D(32, (5,5), activation='relu', input_shape=MNIST_SHAPE)) # 24 x 24
  model.add(MaxPooling2D()) # 12 x 12
  model.add(Conv2D(64, (3,3), activation='relu')) # 10 x 10
  model.add(MaxPooling2D()) # 5 x 5
  model.add(Conv2D(128, (3,3), activation='relu')) # 3 x 3
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(10, activation='sigmoid'))
  model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  return model

# def instantiate_network():
#   model = fashion_network()

def main():
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  # x_hash = make_hash(y_train)
  # labels, pairs = create_pairs(x_hash)
  x_train = np.array(x_train)
  x_test = np.array(x_test)

  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)

  img_rows, img_cols = 28, 28
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255

  # x is image, y is label
  model = fashion_network()
  print(model.summary())
  model.fit(x_train, y_train, verbose=1,
            epochs=3, batch_size=32,
            validation_data=(x_test, y_test))

  score = model.evaluate(x_test, y_test, batch_size=128)
  print('loss: %f \t accuracy: %f' % tuple(score))

if __name__ == '__main__':
  main()