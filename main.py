import cv2
import numpy as np
import sys
import os
import helper
import tensorflow as tf
import train_models_helper
from sklearn.model_selection import train_test_split

EPOCHS = 5
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


def preprocess_images(images):
    new_images = []
    count = 1
    for img in images:
        # Code to change image to grayscale for preprocessing process
        preprocessed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        preprocessed_image = helper.calculate_clahe(preprocessed_image)
        preprocessed_image = helper.image_cut(preprocessed_image)
        if len(preprocessed_image) == 0:
            preprocessed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            preprocessed_image = helper.calculate_clahe(preprocessed_image)
            preprocessed_image = helper.image_cut(preprocessed_image, False, True)
        count += 1
        print(count)
        preprocessed_image = helper.resize_image(preprocessed_image)

        new_images.append(preprocessed_image)

    return new_images


def main():
    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    images = preprocess_images(images)
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    train_models_helper.train_and_test_proposed_model(x_train, x_test, y_train, y_test)


    train_models_helper.train_and_test_alexnet_model(x_train, x_test, y_train, y_test)

    # train without transfer learning
    train_models_helper.train_and_test_pretrained_vgg16_model(x_train, x_test, y_train, y_test)

    # train with transfer learning
    train_models_helper.train_and_test_pretrained_vgg16_model(x_train, x_test, y_train, y_test, transfer_learning=True)


    # train without transfer learning
    train_models_helper.train_and_test_pretrained_vgg19_model(x_train, x_test, y_train, y_test)

    # train with transfer learning
    train_models_helper.train_and_test_pretrained_vgg19_model(x_train, x_test, y_train, y_test, transfer_learning=True)




if __name__ == "__main__":
    main()