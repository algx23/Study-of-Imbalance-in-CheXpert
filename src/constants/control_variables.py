"""Holds the constants that act as controls for an experiment -
things that do not change - ie labels, num epochs, the name
"""

from experiment_utils.arg_parser import parse_arguments

LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
]

MODEL_NAME, *VARS_FOR_EXPERIMENT = parse_arguments()

NUM_EPOCHS = 100
CHEXPERT_COMP_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
