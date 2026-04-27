# Automatic Classification of X-Ray Images

## Information about this repository

This repository holds the code used to create the experimental framework used for the ablations as well as the FINAL RESULTS folder, containing all the files used for the results section in the report.

## Installation Instructions

- To use this repository, follow:
  1. Clone this repository
  2. Download CheXpert-V1.0-Small here: https://www.kaggle.com/datasets/ashery/chexpert/data
  3. Extract the downloaded folder and make a note of the path to "CheXpert-v1.0-small"
  4. In constant/paths.py, change the following:
     - `IMAGES_PATH` to the full path of the parent folder, of where you extracted CheXpert-v1.0-small to - for example, if the path to CheXpert-v1.0-small is in D:/dataset/CheXpert-v1.0-small, `IMAGES_PATH` is set to "D:/dataset/"
     - `ORIGINAL_DATASET_PATH` to the filepath to the train.csv, in the extracted CheXpert-v1.0-small folder, for example: "D:/dataset/CheXpert-v1.0-small/train.csv". By default there is a train.csv in the source folder, so change this if you would like to move it somewhere else

## Running Experiments:

Ensure you are in the src/ folder

- Run pip install -r requirements.txt
  To train a model run
  `py main.py --experiment_root <name> --name <modelname> --args`
- An explanation of the arguments possible can be found by running py main.py --help

- To generate metrics such as the classification report, and other graphs, run:
  `py run_metric_calculator.py --experiment_root <name> --name <modelname>`
  **Ensure that you have already trained the model using main.py before running run_metric_calculator**

- To Run Single Image Inference and GradCam:
  - You Can Either: Run `py predict_single_image.py`, or `py gradcam.py` and follow the on-screen instructions
  - Or you can run `py gradio_ui.py` and click the link that comes up in the terminal and use a simple, prototype user interface.

## Colab or CPU

Each Experiment can be run either on colab, or on a cpu

If running on Colab, please ensure that you select the T4 Runtime.

When running on colab, the commands to run experiments are the same, though `py` must be replaced with `!python`

If running on colab, you can upload the Zip folder of the CheXpert Dataset to your google drive. If you would like to do this, you will also need to zip the src/ folder, and upload it to the colab workspace, using the upward arrow on the left, in the colab "files" tab.

If on colab, please paste the following into the first cell:

```
from google.colab import drive
drive.mount('/content/drive')

!cp /content/drive/MyDrive/archive.zip /content/

!unzip "src.zip"
!unzip archive.zip

!mkdir CheXpert-v1.0-small
!mv train /content/CheXpert-v1.0-small
!mv valid /content/CheXpert-v1.0-small
%cd src/
!pip install -r requirements.txt
```

^ The above assumes that you use the train.csv in the cloned version of this repository, outside the src folder, if not, change `ORIGINAL_DATASET_PATH` as stated above.
