import os
import sys
import cv2
import helper
import csv


def get_dict_data_from_train_csv():
    info_dict = dict()
    with open("train.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        index_of_indicating_cancer = header.index('cancer')
        index_of_patient_id = header.index('image_id')
        for row in csvreader:
            info_dict[row[index_of_patient_id]] = row[index_of_indicating_cancer]

    return info_dict


def load_data_rsna(data_dir: str, images_dict):
    count = 0
    sub_dirs = os.listdir(data_dir)
    images = []
    labels = []
    image_ids = []

    for _, sub in enumerate(sub_dirs):
        pictures_name = os.listdir(os.path.join(data_dir, sub))
        for picture_name in pictures_name:
            filename, _ = os.path.splitext(picture_name)
            img = cv2.imread(os.path.join(data_dir, sub, picture_name))

            images.append(img)
            labels.append(images_dict[filename])
            image_ids.append(filename)
            count += 1 
            print(count)

            
    return (images, labels, image_ids)


def preprocess_images_rsna(images, labels, image_ids):
    new_images = []
    broken_images = []
    count = 0
    for img in images:
        # Code to change image to grayscale for preprocessing process
        preprocessed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        preprocessed_image = helper.calculate_clahe(preprocessed_image)
        preprocessed_image = helper.image_cut(preprocessed_image)
        if not type(preprocessed_image) == list:
            preprocessed_image = helper.resize_image(preprocessed_image)
            broken_images.append(image_ids[count])
            if len(preprocessed_image) == 0:
                preprocessed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                preprocessed_image = helper.calculate_clahe(preprocessed_image)
                preprocessed_image = helper.image_cut(preprocessed_image, False, True)
            new_images.append(preprocessed_image)
        else:
            if labels[count] == '1':
                broken_images.append(image_ids[count])

        count += 1
        print(count)

    return new_images, broken_images


def split_data_into_benign_malignant_rsna(images, labels, image_ids):
    count = 0
    cwd = os.getcwd()

    final_directory_benign = os.path.join(cwd, r'Dataset', r'RSNA', r'Benign Masses')
    final_directory_malignt =  os.path.join(cwd, r'Dataset', r'RSNA', r'Malignant Masses')

    if not os.path.exists(final_directory_benign) and not os.path.exists(final_directory_malignt):
        os.makedirs(final_directory_benign)
        os.makedirs(final_directory_malignt)

    for each_image in images:
        if str(labels[count]) == "0":
            cv2.imwrite(os.path.join(cwd, f"rsna_preprocessed/Benign Masses/{image_ids[count]}.png"), each_image)
        elif str(labels[count]) == "1":
            cv2.imwrite(os.path.join(cwd, f"rsna_preprocessed/Malignant Masses/{image_ids[count]}.png"), each_image)
        count += 1


def main():
    # get the dict train.csv that describes rsna dataset
    images_dict = get_dict_data_from_train_csv()
    
    # Load data
    images, labels, image_ids = load_data_rsna(sys.argv[1], images_dict)

    # preprocess images
    images, broken_images = preprocess_images_rsna(images, labels, image_ids)
    for each in broken_images:
        print(each)

    # Split preprocessed image into two types based on their labels
    split_data_into_benign_malignant_rsna(images, labels, image_ids)



main()
