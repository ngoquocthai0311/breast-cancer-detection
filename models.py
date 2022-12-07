import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

IMG_WIDTH = 227
IMG_HEIGHT = 227
NUM_CATEGORIES = 2
BASE_LEARNING_RATE = 0.0001

def get_proposed_model(IMG_WIDTH = 227, IMG_HEIGHT = 227, NUM_CATEGORIES = 2):

    model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    tf.keras.layers.Rescaling(scale=1./127.5, offset=-1),
    tf.keras.layers.Normalization(),
    tf.keras.layers.Conv2D(8, (3, 3), strides=1, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"),
    tf.keras.layers.Conv2D(16, (3, 3), strides=1, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"),
    tf.keras.layers.Conv2D(32, (3, 3), strides=2, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
    ])
    model
    # Compile model using default settings
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    return model

def get_alexnext_model():
    #Instantiation
    AlexNet = Sequential()

    #1st Convolutional Layer
    AlexNet.add(Conv2D(filters=96, input_shape=(IMG_WIDTH,IMG_HEIGHT,1), kernel_size=(11,11), strides=(4,4), padding='same'))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #2nd Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #3rd Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(Activation('relu'))

    #4th Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(Activation('relu'))

    #5th Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #Passing it to a Fully Connected layer
    AlexNet.add(Flatten())
    # 1st Fully Connected Layer
    AlexNet.add(Dense(4096, input_shape=(32,32,3,)))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    AlexNet.add(Dropout(0.4))

    #2nd Fully Connected Layer
    AlexNet.add(Dense(4096))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #3rd Fully Connected Layer
    AlexNet.add(Dense(1000))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #Output Layer
    AlexNet.add(Dense(NUM_CATEGORIES))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(Activation('softmax'))

    AlexNet.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    #Model Summary
    AlexNet.summary()
    return AlexNet
    

def get_pretrained_vgg16(transfer_learning = False):
    model = VGG16(include_top=False, input_shape=(227, 227, 3),  weights='imagenet')
    # add new classifier layers
    if transfer_learning:
        for layer in model.layers:
            layer.trainable = False
    flat1 = Flatten()(model.layers[-1].output)
    output = Dense(2, activation='softmax')(flat1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # summarize
    model.summary()
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def get_pretrained_vgg19(transfer_learning = False):
    model = VGG19(include_top=False, input_shape=(227, 227, 3),  weights='imagenet')
    if transfer_learning:
        for layer in model.layers:
            layer.trainable = False

    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    output = Dense(2, activation='softmax')(flat1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # summarize
    model.summary()
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model