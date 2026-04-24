"""Holds the paths of the csvs, images, and locations to which metrics
will be saved
"""

from pathlib import Path


IMAGES_PATH = "/content/"
#IMAGES_PATH = "D:/dataset fyp/"

# paths for saving things
CSV_PATHS = Path(f"data/")
ORIGINAL_DATASET_PATH = "../train.csv"
SUBSET_PATH = CSV_PATHS / "subset.csv"
TRAIN_SET_PATH = CSV_PATHS / "train.csv"
VALIDATION_SET_PATH = CSV_PATHS / "validation.csv"
TEST_SET_PATH = CSV_PATHS / "prepared_test.csv"
MEAN_STD_PATH = "norm_const.json"


# def setup_folders(
#     CSV_PATHS,
#     EXPERIMENT_ROOT,
#     MODEL_ROOT,
#     TRAIN_DATA_PATH,
#     VALIDATION_DATA_PATH,
#     EVAL_DATA_PATH,
#     MATRIX_PATH,
#     COMPARISON_PATH,
# ):
#     """Creates folders to which csvs, and metrics will be saved"""

#     CSV_PATHS.mkdir(exist_ok=True, parents=True)
#     EXPERIMENT_ROOT.mkdir(exist_ok=True, parents=True)
#     MODEL_ROOT.mkdir(exist_ok=True, parents=True)
#     TRAIN_DATA_PATH.mkdir(exist_ok=True, parents=True)
#     VALIDATION_DATA_PATH.mkdir(exist_ok=True, parents=True)
#     EVAL_DATA_PATH.mkdir(exist_ok=True, parents=True)
#     MATRIX_PATH.mkdir(exist_ok=True, parents=True)
#     COMPARISON_PATH.mkdir(exist_ok=True, parents=True)
# return
