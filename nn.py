from wrangledata import get_train_data

from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv3D, Dense, Concatenate, Input, MaxPooling2D, Flatten

# 60-64 kernel convolutions

def create_model(image_a, image_b):

    # Build the network shared between both input images
    input_image = Input(shape=image_a.shape)

    # so this 9,9 shape won't work for 3D, it needs to be 3D..., caused errors:
    x = Conv3D(64, (9, 9, 9), padding='same', activation='relu')(input_image)
    # ^ the Conv3D ...(input_image) gives this error:
    # "Input 0 is incompatible with layer conv3d_3: expected ndim=5, found ndim=4"
    # simple question: why are we calling the layers on these tensors?


    x = Conv2D(64, (9, 9), padding='same', activation='relu')(input_image)
    # This doesnt cause an error^^


    x = MaxPooling2D()(x)

    # So the shape param needs to be 3D...
    # Or we are reading in images wrong, in ingest_data.py
    x = Conv3D(128, (7, 7), padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv3D(128, (7, 7), padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    features = Dense(1024, activation='sigmoid')(x)
    shared_model = Model(input_image, features)


    # Why were we passing through the tensors above?


    # Build the feature sets outputed by each network
    a_features = shared_model(image_a)
    b_features = shared_model(image_b)

    # Concatenate the features into one vector for logistic regression
    merged_features = Concatenate([a_features, b_features], axis=-1)

    # Why is logistic regression happening here?
    # Run logistic regression on the image's outputed features
    predictions = Dense(1, activation='sigmoid')(merged_features)

    # Make the model from the inputs and flow of the output
    model = Model(imputs=[image_a, image_b], outputs=predictions)

    # Compile the model. It should be ready to train
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


def create_sub_network(input_shape):
# Need to know the shape it will be X x Y x 3, for the colors...
# Initially we can resize these...



def train_model(model, epochs, data_path):


    model.fit([image_a, image_b], labels, epochs=epochs)