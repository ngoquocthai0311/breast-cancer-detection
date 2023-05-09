import os
import sys
import cv2
import helper


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


def split_data_into_benign_malignant(images, labels):
    count = 0
    cwd = os.getcwd()
    helper.display_img_in_bulk(img=images, time=27)
    for each_image in images:
        if str(labels[count]) == "0":
            cv2.imwrite(os.path.join(cwd, f"processed/Benign Masses/{count}.png"), each_image)
        elif str(labels[count]) == "1":
            cv2.imwrite(os.path.join(cwd, f"processed/Malignant Masses/{count}.png"), each_image)
        count += 1


def main():
    images, labels = load_data(sys.argv[1])
    images = preprocess_images(images)
    split_data_into_benign_malignant(images, labels)
