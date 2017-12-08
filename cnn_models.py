from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout


def three_convolutions(inputShape):
    """
    smaller CNN model for use w/o GPU
    """
    model = Sequential()

    # input 28x28
    model.add(Conv2D(16, (5, 5), padding='same', activation='relu', input_shape=inputShape))
    model.add(MaxPooling2D())
    # input 14 x 14
    model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    # input 7 x 7
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    # input 5 x 5
    model.add(Flatten())
    # input 800
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def five_convolutions(inputShape):
    """
    five-layer CNN for use with a CPU
    """
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=inputShape))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
