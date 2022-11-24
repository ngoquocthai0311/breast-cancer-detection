import cv2
import numpy as np
import sys
import os
import helper
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

EPOCHS = 10
IMG_WIDTH = 227
IMG_HEIGHT = 227
NUM_CATEGORIES = 2
TEST_SIZE = 0.2
Benign_Masses = 0
Malignant_Masses = 1

def load_data(data_dir):
    count = 0
    sub_dirs = os.listdir(data_dir)
    images = []
    labels = []

    # Get the images from the folder
    for sub_dir in sub_dirs:
        picture_paths = os.listdir(os.path.join(data_dir, sub_dir))
        for each_picture_path in picture_paths:
            image_folder = os.listdir(os.path.join(data_dir, sub_dir, each_picture_path))
            label = None
            if each_picture_path == "Benign-Masses":
                label = Benign_Masses
            else:
                label = Malignant_Masses
            # print(image_folder)
            for each_picture in image_folder:
                # retrieve image
                img = cv2.imread(os.path.join(data_dir, sub_dir, each_picture_path, each_picture))

                images.append(img)
                labels.append(label)
                count += 1
                print(count)

    return (images, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        # Input layer without fitering
        tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        tf.keras.layers.Rescaling(scale=1./127.5, offset=-1),
        tf.keras.layers.Normalization(),
        # Convolutional layer. Learn 32 filters using a 3x3 kernel
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
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Flatten(),
        # # Hidden layer with dropout rate by 0.5
        # tf.keras.layers.Dense(128, activation="relu"),
        # tf.keras.layers.Dropout(0.5),   

        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
    ])

    # Compile model using default settings
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    return model


def preprocess_images(images):
    count = 1
    broken = 0
    broken_index = dict()
    new_images = []
    for img in images:
        # Code to change image to grayscale for preprocessing process
        preprocessed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        preprocessed_image = helper.calculate_clahe(preprocessed_image)
        preprocessed_image = helper.image_cut(preprocessed_image)
        preprocessed_image = helper.resize_image(preprocessed_image)

        # preprocessed_image = tf.image.per_image_standardization(preprocessed_image)

        new_images.append(preprocessed_image)
        # print(type(preprocessed_image))
        # preprocessed_image = None
        # print(preprocessed_image)

        # display_img_in_bulk(preprocessed_image)
        
        if len(preprocessed_image) == 0:
            broken += 1
            broken_index[count - 1] = img
        count += 1
        # cv2.imshow("normal image", img)
        # cv2.imshow("after preprocessed", preprocessed_image)
        # cv2.waitKey(0)
    print('broken = ', broken)
    for key in broken_index.keys():
        preprocessed_image = cv2.cvtColor(broken_index[key], cv2.COLOR_BGR2GRAY)
        preprocessed_image = helper.calculate_clahe(preprocessed_image)
        preprocessed_image = helper.image_cut(preprocessed_image, True)

    return new_images

def display_img_in_bulk(img):
    cv2.imshow('Window', img)

    key = cv2.waitKey(50)
    if key == 27:#if ESC is pressed, exit loop
        cv2.destroyAllWindows()


def main():

     # Check command-line arguments
    # if len(sys.argv) not in [2, 3]:
    #     sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    images = preprocess_images(images)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # # ref: https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/
    # # reshape dataset to have a single channel
    # width, height, channels = x_train.shape[1], x_train.shape[2], 1
    # x_train = x_train.reshape((x_train.shape[0], width, height, channels))
    # x_test = x_test.reshape((x_test.shape[0], width, height, channels))
    # # create generator to center images
    # datagen = ImageDataGenerator(rescale=1.0/255.0 ,featurewise_center=True)
    # # calculate mean on training dataset
    # datagen.fit(x_train)
    # # prepare an iterators to scale images
    # train_iterator = datagen.flow(x_train, y_train)
    # test_iterator = datagen.flow(x_test, y_test)

    # Get a compiled neural network
    model = get_model()

    # print(train_iterator[0])
    # print(train_iterator[1])    
    # Old code
    # Fit model on training data
    # model.fit(train_iterator[0], train_iterator[1], epochs=EPOCHS)

    # Evaluate neural network performance
    # model.evaluate(test_iterator[0], test_iterator[1], verbose=2)

    # Fit model on training data
    # model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=EPOCHS)

    # # Evaluate neural network performance
    # model.evaluate_generator(test_iterator,  steps=len(test_iterator), verbose=2)

    # Old code
    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # # Save model to file
    # if len(sys.argv) == 3:
    #     filename = sys.argv[2]
    #     model.save(filename)
    #     print(f"Model saved to {filename}.")


if __name__ == "__main__":
    main()