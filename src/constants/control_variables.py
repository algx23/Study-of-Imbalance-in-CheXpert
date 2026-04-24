"""Holds the constants that act as controls for an experiment -
things that do not change - ie labels, num epochs, the name
"""

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


NUM_EPOCHS = 100
CHEXPERT_COMP_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
RARE_CHEXPERT_LABELS = [
    "Fracture",
    "Pleural Other",
    "Pneumothorax",
    "Lung Lesion",
    "Enlarged Cardiomediastinum",
    "Pneumonia",
]
