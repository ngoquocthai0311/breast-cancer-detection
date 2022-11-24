## Breast Cancer Detection using Artificial Intelligence

- This is an attempt in implementing the AI model that will be used to detect first stage breast cancer. The datasets or images are collected and used in purely educational purposes. 

## How to get the dataset

- The dataset is downloaded via this link: https://data.mendeley.com/datasets/ywsbh3ndr8/5

- After having downloaded and extracted at the code base folder, the image folders must be renamed as they do not have spaces between words.

## How to run the code

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
