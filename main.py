import cv2
import numpy as np
import sys
import os
import helper
import tensorflow as tf
import train_models_helper
from split_folder_into_train_and_val import split_preprocessed_images_to_train_and_val_folder
import shutil


def load_data(data_dir):
    train_folder, val_folder = os.listdir(data_dir)
    images_train = []
    labels_train  = []
    images_test = []
    labels_test = []
    # Get train image
    for index, each_sub in enumerate(os.listdir(os.path.join(data_dir, train_folder))):
        for image in os.listdir(os.path.join(data_dir, train_folder, each_sub)):
            img = cv2.imread(os.path.join(data_dir, train_folder, each_sub, image))
            preprocessed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            preprocessed_image = helper.resize_image(preprocessed_image)
            images_train.append(preprocessed_image)
            labels_train.append(index)

    # Get test image
    for index, each_sub in enumerate(os.listdir(os.path.join(data_dir, val_folder))):
        for image in os.listdir(os.path.join(data_dir, val_folder, each_sub)):
            img = cv2.imread(os.path.join(data_dir, val_folder, each_sub, image))
            preprocessed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            preprocessed_image = helper.resize_image(preprocessed_image)
            images_test.append(preprocessed_image)
            labels_test.append(index)
    return images_train, labels_train, images_test, labels_test


def main():
    dataset_name = os.path.basename(os.path.dirname(sys.argv[1]))

    # Randomly split preprocessed images into training and validating folder
    output_split = split_preprocessed_images_to_train_and_val_folder(dataset_name)

    # Get image arrays and labels for all image files
    images_train, labels_train, images_test, labels_test = load_data(os.path.join(os.getcwd(), output_split))
    x_train = np.array(images_train)
    x_test = np.array(images_test)
    y_train = np.array(tf.keras.utils.to_categorical(labels_train))
    y_test = np.array(tf.keras.utils.to_categorical(labels_test))

    # Train and validate latest proposed model
    train_models_helper.train_and_test_proposed_model(x_train, x_test, y_train, y_test)

    # # Train and validate alexnet model
    # train_models_helper.train_and_test_alexnet_model(x_train, x_test, y_train, y_test)

    # # Train and validate VGG16 model without applying transfer learning method
    # train_models_helper.train_and_test_pretrained_vgg16_model(x_train, x_test, y_train, y_test)

    # # Train and validate alexnet model with applied transfer learning method
    # train_models_helper.train_and_test_pretrained_vgg16_model(x_train, x_test, y_train, y_test, transfer_learning=True)

    # # Train and validate alexnet model without applying transfer learning method
    # train_models_helper.train_and_test_pretrained_vgg19_model(x_train, x_test, y_train, y_test)

    # # Train and validate alexnet model with applied transfer learning method
    # train_models_helper.train_and_test_pretrained_vgg19_model(x_train, x_test, y_train, y_test, transfer_learning=True)

if __name__ == "__main__":
    previous_random_split_folder = os.path.join(os.getcwd(), r'output')
    if os.path.exists(previous_random_split_folder):
        shutil.rmtree(previous_random_split_folder)
    main()