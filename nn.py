from wrangledata import get_train_data

from keras.models import Model, Sequential
from keras.layers import Conv3D, Dense, Concatenate, Input, MaxPooling2D, Flatten

# 60-64 kernel convolutions

def create_model(image_a, image_b):

    # Build the network shared between both input images
    input_image = Input(shape=image_a.shape)
    x = Conv3D(64, (9, 9), padding='same', activation='relu')(input_image)
    x = MaxPooling2D()(x)
    x = Conv3D(128, (7, 7), padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv3D(128, (7, 7), padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    features = Dense(1024, activation='sigmoid')(x)
    shared_model = Model(input_image, features)

    # Build the feature sets outputed by each network
    a_features = shared_model(image_a)
    b_features = shared_model(image_b)

    # Concatenate the features into one vector for logistic regression
    merged_features = Concatenate([a_features, b_features], axis=-1)

    # Run logistic regression on the image's outputed features
    predictions = Dense(1, activation='sigmoid')(merged_features)

    # Make the model from the inputs and flow of the output
    model = Model(imputs=[image_a, image_b], outputs=predictions)

    # Compile the model. It should be ready to train
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])




def train_model(model, epochs, data_path):



    model.fit([image_a, image_b], labels, epochs=epochs)