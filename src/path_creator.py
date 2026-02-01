from constants import MODEL_NAME
from pathlib import Path


MODEL_ROOT = Path(f"results/{MODEL_NAME}")
TRAIN_DATA_PATH = Path(f"{MODEL_ROOT}/train")

VALIDATION_DATA_PATH = Path(f"{MODEL_ROOT}/validation")

EVAL_DATA_PATH = Path(f"{MODEL_ROOT}/evaluation")
MATRIX_PATH = Path(f"{EVAL_DATA_PATH}/confusion matrices")

def setup_folders():

    MODEL_ROOT.mkdir(exist_ok=True, parents=True)
    TRAIN_DATA_PATH.mkdir(exist_ok=True, parents=True)
    VALIDATION_DATA_PATH.mkdir(exist_ok=True, parents=True)
    EVAL_DATA_PATH.mkdir(exist_ok=True, parents=True)
    MATRIX_PATH.mkdir(exist_ok=True, parents=True)
    return
