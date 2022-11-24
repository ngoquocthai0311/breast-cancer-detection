import cv2
import numpy as np
import sys
import os
import helper
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 227
IMG_HEIGHT = 227
NUM_CATEGORIES = 2
TEST_SIZE = 0.2
Benign_Masses = 0
Malignant_Masses = 1

def load_data(data_dir):

    sub_dirs = os.listdir(data_dir)
    images = []
    labels = []

    for index,sub in enumerate(sub_dirs):
        pictures_name = os.listdir(os.path.join(data_dir, sub))
        for picture_name in pictures_name:
            label = index
            # retrieve image
            img = cv2.imread(os.path.join(data_dir, sub, picture_name))

            images.append(img)
            labels.append(label)
            

    return (images, labels)

def get_model():
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

        new_images.append(preprocessed_image)
        
        if len(preprocessed_image) == 0:
            broken += 1
            broken_index[count - 1] = img
        count += 1
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

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    images = preprocess_images(images)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)


if __name__ == "__main__":
    main()