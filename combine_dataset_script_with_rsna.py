import os
import sys
import cv2
import helper
import csv
import random

def load_dataset(data_dir, rsna_dataset):
    sub_dirs = os.listdir(data_dir)
    images = []
    labels = []

    # load first dataset
    for index,sub in enumerate(sub_dirs):
        pictures_name = os.listdir(os.path.join(data_dir, sub))
        for picture_name in pictures_name:
            label = index
            # retrieve image
            img = cv2.imread(os.path.join(data_dir, sub, picture_name))

            images.append(img)
            labels.append(label)

    # load rsna dataset malignant only
    rsna_malignant_pictures_name = os.listdir(os.path.join(rsna_dataset, r'Malignant Masses'))
    length_of_rsna_malignant_dataset = 0
    for picture_name in rsna_malignant_pictures_name:
        # retrieve image
        img = cv2.imread(os.path.join(rsna_dataset, r'Malignant Masses', picture_name))

        images.append(img)
        labels.append(1)
        length_of_rsna_malignant_dataset += 1 
    
    # load rsna dataset benign only
    rsna_benign_pictures_name = os.listdir(os.path.join(rsna_dataset, r'Benign Masses'))
    random_rsna_benign_pictures_name = random.sample(rsna_benign_pictures_name, length_of_rsna_malignant_dataset)
    for picture_name in random_rsna_benign_pictures_name:
        # retrieve image
        img = cv2.imread(os.path.join(rsna_dataset, r'Benign Masses', picture_name))

        images.append(img)
        labels.append(0)
            

    return (images, labels)

def write_new_collect_images(images, labels):
    count = 0
    cwd = os.getcwd()

    final_directory_benign = os.path.join(cwd, r'Dataset', r'INbreast+MIAS+DDSM+RSNA', r'Benign Masses')
    final_directory_malignt =  os.path.join(cwd, r'Dataset', r'INbreast+MIAS+DDSM+RSNA', r'Malignant Masses')

    if not os.path.exists(final_directory_benign) and not os.path.exists(final_directory_malignt):
        os.makedirs(final_directory_benign)
        os.makedirs(final_directory_malignt)

    for each_image in images:
        if labels[count] == 0:
            cv2.imwrite(os.path.join(cwd, f"Dataset/INbreast+MIAS+DDSM+RSNA/Benign Masses/{count}.png"), each_image)
        elif str(labels[count]) == "1":
            cv2.imwrite(os.path.join(cwd, f"Dataset/INbreast+MIAS+DDSM+RSNA/Malignant Masses/{count}.png"), each_image)
        count += 1


def main():
    # load both dataset
    images, labels = load_dataset(sys.argv[1], sys.argv[2])
    write_new_collect_images(images, labels)



main()
