from wrangledata import get_train_data
from PIL import Image

from keras.backend import variable as MakeTensor
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv3D, Dense, Concatenate, Input, MaxPooling2D, Flatten
from sys import argv

# 60-64 kernel convolutions

def image_to_tensor(path):
    return MakeTensor(Image.open(path))

def create_model(image_a, image_b):

    input_shape = image_a.shape()
    sub_net = create_sub_network(input_shape)

    input_a = Input(input_shape)
    input_b = Input(input_shape)

    # Build the feature sets outputed by each network
    features_a = sub_net(input_a)
    features_b = sub_net(input_b)

    # Concatenate the features into one vector for logistic regression
    merged_features = Concatenate([features_a, features_b], axis=-1)

    # Logistic regression in place of a distance function.
    # Run logistic regression on the image's outputed features
    predictions = Dense(1, activation='sigmoid')(merged_features)

    # Make the model from the inputs and flow of the output
    model = Model(imputs=[image_a, image_b], outputs=predictions)

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



def train_model(model, epochs, data_path):

    # USE flow_from_directory() from this page: https://keras.io/preprocessing/image/


    model.fit([image_a, image_b], labels, epochs=epochs)

def main():
    image_a = argv[1]
    image_b = argv[2]
    create_model(image_a, image_b)

if __name__ == '__main__':
    main()