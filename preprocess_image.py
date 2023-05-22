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


def split_data_into_benign_malignant(images, labels, base_folder_name):
    count = 0
    cwd = os.getcwd()
    final_directory_benign = os.path.join(cwd, r'Dataset', base_folder_name, r'Benign Masses')
    final_directory_malignt =  os.path.join(cwd, r'Dataset', base_folder_name, r'Malignant Masses')
    if not os.path.exists(final_directory_benign) and not os.path.exists(final_directory_malignt):
        os.makedirs(final_directory_benign)
        os.makedirs(final_directory_malignt)
    for each_image in images:
        if str(labels[count]) == "0":
            cv2.imwrite(os.path.join(cwd, f"Dataset/{base_folder_name}/Benign Masses/{count}.png"), each_image)
        elif str(labels[count]) == "1":
            cv2.imwrite(os.path.join(cwd, f"Dataset/{base_folder_name}/Malignant Masses/{count}.png"), each_image)
        count += 1


def main():
    # Get dataset name
    dataset_name = os.path.basename(os.path.dirname(sys.argv[1]))

    # Load data
    images, labels = load_data(sys.argv[1])

    # Preprocessed images
    images = preprocess_images(images)

    # Split preprocessed image into two types based on their labels
    split_data_into_benign_malignant(images, labels, dataset_name)

main()
