"""Holds the paths of the csvs, images, and locations to which metrics
will be saved
"""

from pathlib import Path


# IMAGES_PATH = "/content/"
IMAGES_PATH = "D:/dataset fyp/"

# paths for saving things
CSV_PATHS = Path(f"data/")
ORIGINAL_DATASET_PATH = "../train.csv"
SUBSET_PATH = CSV_PATHS / "subset.csv"
TRAIN_SET_PATH = CSV_PATHS / "train.csv"
VALIDATION_SET_PATH = CSV_PATHS / "validation.csv"
TEST_SET_PATH = CSV_PATHS / "prepared_test.csv"
MEAN_STD_PATH = "norm_const.json"
