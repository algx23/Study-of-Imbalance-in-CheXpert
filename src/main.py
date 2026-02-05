import os
import pandas as pd
from datetime import datetime
from experiment_utils.arg_parser import parse_arguments

from data_preparation import DataSubsetter, prepare_data, prepare_test_data

import torch
from chexpert_dataset import ChexpertDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import Compose,Normalize, Resize, ToTensor # to resize all images
from utils import (calculate_mean_and_standard_deviation,
                   calculate_class_weights,
                   save_model,
                   write_train_loss_to_file)


from model import BaselineModel
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from constants import (
    NUM_EPOCHS,
    LABELS,
    MODEL_NAME,
    IMAGES_PATH,
    ORIGINAL_DATASET_PATH,
    SUBSET_PATH,
    TRAIN_SET_PATH,
    VALIDATION_SET_PATH,
    TEST_SET_PATH
)
from validation_loss_checker import ValidationLossChecker

import matplotlib.pyplot as plt
from utils import plot_loss

from pathlib import Path
from path_creator import (
    setup_folders,
    MODEL_ROOT,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    EVAL_DATA_PATH,
    MATRIX_PATH
)

from sklearn.metrics import (multilabel_confusion_matrix,
                             classification_report,
                             roc_auc_score,
                             ConfusionMatrixDisplay)
import numpy as np
from PIL import Image

# augmentation imports
from torchvision.transforms import (RandomRotation,
                                    RandomHorizontalFlip)


from trainer import Trainer
from evaluate import EvaluationLoop

if __name__ == "__main__":
    setup_folders()

    _, augment_transforms, use_weights, use_clahe, use_dropout, use_batch_norm = parse_arguments() 

    # make the parent folder all of the logs, images, model will go into
    print(f"Evaluating Model {MODEL_NAME}")
    print(f"USE DROPOUT: {use_dropout}")
    print(f"CLASS WEIGHTS USED {use_weights}")
    print(f"BATCH NORM USED:  {use_batch_norm}")
    print(f"START TIME {datetime.now()}")

    subsetter = DataSubsetter(ORIGINAL_DATASET_PATH, TRAIN_SET_PATH, VALIDATION_SET_PATH, TEST_SET_PATH, LABELS)

    if not (os.path.exists(TRAIN_SET_PATH)
            and os.path.exists(VALIDATION_SET_PATH)):
        print("subsetting data to create train and validation files")
        subsetter.create_subset_train_validation()
    else:
        print("Train / Valid Subsets already created. Loader prep initializing..")

    if not os.path.exists(TEST_SET_PATH):
        print("creating test dataset now")
        subsetter.create_test_data_csv()

    dataloader_for_training, data_loader_for_validation = prepare_data(augment_transforms, use_clahe, IMAGES_PATH, TRAIN_SET_PATH, VALIDATION_SET_PATH)
    data_loader_for_testing = prepare_test_data(TEST_SET_PATH)

    class_weights = calculate_class_weights(TRAIN_SET_PATH) if use_weights else None
    

    if not os.path.exists(f'{MODEL_ROOT}/{MODEL_NAME}.pt'):
        model = BaselineModel(use_dropout=use_dropout, use_batch_norm=use_batch_norm)
        optimizer = Adam(model.parameters(), lr=1e-4)
        loss_fn = BCEWithLogitsLoss(pos_weight=class_weights)

        print("no previous models, training now")
        trainer = Trainer(model, optimizer=optimizer, loss_fn=loss_fn, train_loader=dataloader_for_training, validation_loader=data_loader_for_validation, class_weights=class_weights, NUM_EPOCHS=NUM_EPOCHS)
        train_losses, validation_losses, epochs = trainer.train_model()
        plot_loss(train_losses, validation_losses, epochs)
    else:
        print("previous models found!")
        model = torch.load(MODEL_ROOT / f"{MODEL_NAME}.pt", weights_only=False)

    post_train_model = model
    print(post_train_model)

    loss_fn = BCEWithLogitsLoss(pos_weight=class_weights)

    print('EVALUATION STARTING')
    eval_loop = EvaluationLoop(data_loader_for_testing, post_train_model, loss_fn)
    eval_loop.evaluate_model()
    print(f"FINISH TIME {datetime.now()}")

