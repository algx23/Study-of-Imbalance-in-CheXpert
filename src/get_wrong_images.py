from pathlib import Path

import torch

from constants.control_variables import RARE_CHEXPERT_LABELS, LABELS
from constants.paths import MODEL_ROOT, TEST_SET_PATH
import pandas as pd

from data_preparation.loader_prep import prepare_test_data


def get_wrong_images(MODEL_ROOT, TEST_SET_PATH, MODEL_NAME, device):
    model_root = Path(MODEL_ROOT)

    if Path(TEST_SET_PATH).is_file():
        test_loader = prepare_test_data(TEST_SET_PATH)
        exit(1)
    else:
        print(
            "Go to Main.py and please train a model to create the test set path again"
        )

    true_labels_file = model_root / "evaluation" / "truth_tensor.pt"
    predicted_labels_file = model_root / "evaluation" / "prediction_tensor.pt"
    if true_labels_file.is_file() and predicted_labels_file.is_file():

        images_in_batch = []

        for i, data in enumerate(test_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            images_in_batch.extend(images)

            true_labels = torch.load(true_labels_file)
            predicted_labels = torch.load(predicted_labels_file)
            index_of_wrong_classification = (
                (predicted_labels == true_labels) == False
            ).nonzero()

        # sample_of_images = 1
        # for i in range(len(LABELS)):
        #     index_of_class_to_get_col = LABELS[i].index()
        #     if true_labels[:, i] != predicted_labels[:, i]:
        #         misclassified_image = test_loader.dataset[i]
