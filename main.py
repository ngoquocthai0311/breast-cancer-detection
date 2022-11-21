import cv2
import sys
import os
import numpy as np
import helper
from matplotlib import pyplot as plt

def load_data(data_dir):
    sub_dirs = os.listdir(data_dir)
    images = []
    labels = []

    # Get the images from the folder
    for sub_dir in sub_dirs:
        picture_paths = os.listdir(os.path.join(data_dir, sub_dir))
        for each_picture_path in picture_paths:
            image_folder = os.listdir(os.path.join(data_dir, sub_dir, each_picture_path))
            for each_picture in image_folder:
                # retrieve image
                img = cv2.imread(os.path.join(data_dir, sub_dir, each_picture_path, each_picture))

                images.append(img)
                labels.append(each_picture_path)

    return (images, labels)


def preprocess_images(images):
    for img in images:
        # Code to change image to grayscale for preprocessing process
        preprocessed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        preprocessed_image = helper.calculate_clahe(preprocessed_image)

        preprocessed_image = helper.image_cut(preprocessed_image)

        helper.plot_image(preprocessed_image)
        # cv2.imshow("normal image", img)
        # cv2.imshow("after preprocessed", preprocessed_image)
        # cv2.waitKey(0)


def main():
    images, labels = load_data(sys.argv[1])
    preprocess_images(images)


if __name__ == "__main__":
    main()