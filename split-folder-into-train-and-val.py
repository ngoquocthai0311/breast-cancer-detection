import splitfolders
import os

# ref https://stackoverflow.com/questions/53074712/how-to-split-folder-of-images-into-test-training-validation-sets-with-stratified

OUTPUT_FOLFDER = "output"

absolute_path = os.path.dirname(__file__)
relative_path_DDSM = "Dataset/DDSM"
relative_path_Inbreast = "Dataset/INbreast"
relative_path_MIAS = "Dataset/MIAS"
relative_path_combineddataset = "Dataset/INbreast+MIAS+DDSM"

full_path_DDSM = os.path.join(absolute_path, relative_path_DDSM)
full_path_INbreast = os.path.join(absolute_path, relative_path_Inbreast)
full_path_MIAS = os.path.join(absolute_path, relative_path_MIAS)
full_path_combineddataset = os.path.join(absolute_path, relative_path_combineddataset)


splitfolders.ratio(full_path_DDSM, output=os.path.join(OUTPUT_FOLFDER, relative_path_DDSM),
    seed=1337, ratio=(.8, .2), group_prefix=None, move=False) # default values

splitfolders.ratio(full_path_INbreast, output=os.path.join(OUTPUT_FOLFDER, relative_path_Inbreast),
    seed=1337, ratio=(.8, .2), group_prefix=None, move=False) # default values

splitfolders.ratio(full_path_MIAS, output=os.path.join(OUTPUT_FOLFDER, relative_path_MIAS),
    seed=1337, ratio=(.8, .2), group_prefix=None, move=False) # default values

splitfolders.ratio(full_path_combineddataset, output=os.path.join(OUTPUT_FOLFDER, relative_path_combineddataset),
    seed=1337, ratio=(.8, .2), group_prefix=None, move=False) # default values