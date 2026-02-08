from constants.control_variables import MODEL_NAME
from pathlib import Path

IMAGES_PATH = "D:/dataset fyp/"
ORIGINAL_DATASET_PATH = "../train.csv"
SUBSET_PATH = "data/subset.csv"
TRAIN_SET_PATH = "data/train.csv"
VALIDATION_SET_PATH = "data/validation.csv"
TEST_SET_PATH = "data/prepared_test.csv"

# paths for saving things
CSV_PATHS = Path(f"data/")

MODEL_ROOT = Path(f"results/{MODEL_NAME}")
TRAIN_DATA_PATH = Path(f"{MODEL_ROOT}/train")

VALIDATION_DATA_PATH = Path(f"{MODEL_ROOT}/validation")

EVAL_DATA_PATH = Path(f"{MODEL_ROOT}/evaluation")
MATRIX_PATH = Path(f"{EVAL_DATA_PATH}/confusion matrices")
COMPARISON_PATH = Path(f"comparisons/")


def setup_folders():

    CSV_PATHS.mkdir(exist_ok=True, parents=True)
    MODEL_ROOT.mkdir(exist_ok=True, parents=True)
    TRAIN_DATA_PATH.mkdir(exist_ok=True, parents=True)
    VALIDATION_DATA_PATH.mkdir(exist_ok=True, parents=True)
    EVAL_DATA_PATH.mkdir(exist_ok=True, parents=True)
    MATRIX_PATH.mkdir(exist_ok=True, parents=True)
    COMPARISON_PATH.mkdir(exist_ok=True, parents=True)
    return
