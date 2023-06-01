import splitfolders
import os
import random
import sys

# ref https://stackoverflow.com/questions/53074712/how-to-split-folder-of-images-into-test-training-validation-sets-with-stratified


# delete previous randomly split dataset
# cwd = os.getcwd()
# os.rmdir(os.path.join(cwd, r'Dataset'))

OUTPUT_FOLFDER = "output"
INPUT_FOLFDER = "preprocessed"

absolute_path = os.path.dirname(__file__)
relative_path_DDSM = "Dataset/DDSM"
relative_path_Inbreast = "Dataset/INbreast"
relative_path_MIAS = "Dataset/MIAS"
relative_path_combineddataset = "Dataset/INbreast+MIAS+DDSM"

full_path_DDSM = os.path.join(absolute_path, relative_path_DDSM)
full_path_INbreast = os.path.join(absolute_path, relative_path_Inbreast)
full_path_MIAS = os.path.join(absolute_path, relative_path_MIAS)
full_path_combineddataset = os.path.join(absolute_path, relative_path_combineddataset)


def split_preprocessed_images_to_train_and_val_folder(dataset_name, train_ratio = .8, validate_ratio = .2, output_folder = OUTPUT_FOLFDER):
    random.seed()
    random_value = random.randint(1, 9999)
    print(random_value)
    
    if dataset_name == "MIAS":
        splitfolders.ratio(full_path_MIAS,
                        output=output_folder,
                        seed=random_value,
                        ratio=(train_ratio, validate_ratio),
                        group_prefix=None,
                        move=False
                        ) # default values
    elif dataset_name == "INbreast":
        splitfolders.ratio(full_path_INbreast,
                    output=output_folder,
                    seed=random_value,
                    ratio=(train_ratio, validate_ratio),
                    group_prefix=None,
                    move=False) # default values
    elif dataset_name == "DDSM":
        splitfolders.ratio(full_path_DDSM,
                       output=output_folder,
                       seed=random_value,
                       ratio=(train_ratio, validate_ratio),
                       group_prefix=None,
                       move=False) # default values
    elif dataset_name == "INbreast+MIAS+DDSM":
        splitfolders.ratio(full_path_combineddataset,
                           output=output_folder,
                           seed=random_value,
                           ratio=(train_ratio, validate_ratio),
                           group_prefix=None,
                           move=False) # default values

def main():
    split_preprocessed_images_to_train_and_val_folder()

main()
