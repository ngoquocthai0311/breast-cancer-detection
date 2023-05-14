## Classification Of Breast Cancer For Modern Mammography Using Convolutional Neural Network

- This is an attempt in implementing the AI model that will be used to detect first stage breast cancer. The datasets or images are collected and used in purely educational purposes. 
- The AI machine learning models used in this repo are latest suggested model in 2022, Alexnet, VGG16-19, and sub repository of HRnet which is HRnet for image classfication.
- In this readme, the written instructions below are for guiding reader to set up the dataset and running the machine learning models.

## Download the dataset

- The dataset is downloaded via this link: https://data.mendeley.com/datasets/ywsbh3ndr8/5

- After having downloaded and extracted at the code base folder, the image folders must be renamed as they does not have spaces between words.

## Processing breast mass images

- To successfully preprocess images from the mendeley dataset, assuming you have downloaded the dataset, please name the dataset folder as your preferences that it must not contain any white spaces or special characters. Then you would safely run the code as below to preprocess image.

- ```python
  python ./preprocess_image.py ./{relative path to your dataset}
  ```

- Please note that to simplify the process and save computing resources, 'preprocess_image.py' script only preprocess specficied dataset, i.e DDSM, MIAS, INbreast, or combined dataset. Please also note that this script can not preprocess the whole downloaded dataset from mendeley.

## About the training and validating model script

- In this thesis, the whole process of running the script to train models will be runned manually by the user. To save the computing resources, please comment out the model that you would not be interested in training. Below is the example to do so.

- The script below is the original implementation in the main.py script

- ```python
      # Train and validate latest proposed model
      train_models_helper.train_and_test_proposed_model(x_train, x_test, y_train, y_test)
  
      # Train and validate alexnet model
      train_models_helper.train_and_test_alexnet_model(x_train, x_test, y_train, y_test)
  
      # Train and validate VGG16 model without applying transfer learning method
      train_models_helper.train_and_test_pretrained_vgg16_model(x_train, x_test, y_train, y_test)
  
      # Train and validate alexnet model with applied transfer learning method
      train_models_helper.train_and_test_pretrained_vgg16_model(x_train, x_test, y_train, y_test, transfer_learning=True)
  
      # Train and validate alexnet model without applying transfer learning method
      train_models_helper.train_and_test_pretrained_vgg19_model(x_train, x_test, y_train, y_test)
  
      # Train and validate alexnet model with applied transfer learning method
      train_models_helper.train_and_test_pretrained_vgg19_model(x_train, x_test, y_train, y_test, transfer_learning=True)
  ```

- If you are only interested in training and validating the latest suggested model, please comment out all the remaining models as follows:

- ```python
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
  ```

- Please note that this script is only responsible for training and validating latest suggested mode, Alexet, VGG16-19 models.

### Training and validating HRnet model.

- In this repo, the implementation for HRnet is not included since the open source is available for cloning. To run the HRnet model for image classification inside this repo. Please downloading or cloning the repo as follow:

- ```bash
  https://github.com/HRNet/HRNet-Image-Classification.git
  ```

- After that, please read the instruction from the HRnet image classification itself to know how to run the model.

- In addition, the spliting folder script is not runned automatically when running HRnet model. Therefore, there is a split_folder_into_train_and_va.py script that are also available to run independently from main.py script to split the preprocessed images.

- ```python
  python ./split_folder_into_train_and_val.py ./{path to specified dataset}
  ```

## Run the main sccript to training and validating machine learning models except HRnet

1. Rename the dataset as suggested.

2. ```
   pip install -r .\requirements.txt
   ```

3. ```python
   python .\main.py .\path\to\dataset
   ```
- The path to the dataset will be assumed that inside the folder will include 2 sub folders that are already categorized into 2 different classifications. The structure of the expected dataset will be looked like this:
  
  - Dataset
    
    - Benign Masses
    
    - Benign Masses

- If you have the suggested dataset structure, please execute the code as:
  
  ```python
  python .\main.py .\Dataset\
  ```
