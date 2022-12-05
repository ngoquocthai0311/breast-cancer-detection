import cv2
import numpy as np
import sys
import os
import helper
import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
# from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from proposed_model import get_alexnext_model, get_model, get_pretrained_vgg16, get_pretrained_vgg19
import create_dataset
import torchvision.models as models

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

            # test_image = yolov4_model.mask_to_yolo(img)
            # display_img_in_bulk(test_image, 500)

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

        # display_img_in_bulk(preprocessed_image, 50)
        # cv2.imshow("normal image", img)
        # cv2.imshow("after preprocessed", preprocessed_image)
        # cv2.waitKey(0)

        new_images.append(preprocessed_image)

    return new_images

def display_img_in_bulk(img, time):
    cv2.imshow('Window', img)

    key = cv2.waitKey(time)
    if key == 27:#if ESC is pressed, exit loop
        cv2.destroyAllWindows()


def main():

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    images = preprocess_images(images)
    rgb_images = helper.convert_to_rgb(images)
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )


    # alexnext_model.fit(x_train, y_train, epochs=EPOCHS)
    # alexnext_model.evaluate(x_test,  y_test, verbose=2)
    # vgg_19_model = get_pretrained_vgg19()
    # # # model2 = get_pretrained_vgg16()
    # # # Fit model on training data
    # vgg_19_model.fit(np.array(helper.convert_to_rgb(x_train)), y_train, epochs=EPOCHS)

    # # Evaluate neural network performance
    # vgg_19_model.evaluate(np.array(helper.convert_to_rgb(x_test)), y_test, verbose=2)

    # vgg_16_model = get_pretrained_vgg16()
    # # model2 = get_pretrained_vgg16()
    # # Fit model on training data
    # vgg_16_model.fit(np.array(helper.convert_to_rgb(x_train)), y_train, epochs=EPOCHS)
    # # vgg_16_model.fit(x_train, y_train, epochs=EPOCHS)
    # # Evaluate neural network performance
    # vgg_16_model.evaluate(np.array(helper.convert_to_rgb(x_test)), y_test, verbose=2)
    # # vgg_16_model.evaluate(x_test,  y_test, verbose=2)

    # model = get_model()
    # # alexnext_model = get_alexnext_model()
    # # alexnext_model = models.alexnet(pretrained=True)

    model = get_model()
    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)




if __name__ == "__main__":
    main()