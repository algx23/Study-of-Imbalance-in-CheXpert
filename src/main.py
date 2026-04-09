import os
from datetime import datetime

import torch

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from constants.control_variables import (
    LABELS,
    MODEL_NAME,
    NUM_EPOCHS,
    VARS_FOR_EXPERIMENT,
)
from constants.paths import (
    IMAGES_PATH,
    MODEL_ROOT,
    ORIGINAL_DATASET_PATH,
    TEST_SET_PATH,
    TRAIN_SET_PATH,
    VALIDATION_SET_PATH,
    setup_folders,
)
from data_preparation import DataSubsetter, prepare_data, prepare_test_data
from evaluate import EvaluationLoop
from model import BaselineModel
from trainer import Trainer
from utils import (
    calculate_class_weights,
    plot_loss,
    calculate_normalized_inverse_frequency_focal_loss,
    calculate_class_freq_cbfl,
)

from comparison_generator import generate_comparisons
from custom_loss_fns.focal_loss import FocalLoss
from custom_loss_fns.class_balanced_focal_loss import ClassBalancedFocalLoss
import json

if __name__ == "__main__":
    augment_transforms, use_weights, use_clahe, use_dropout, use_batch_norm, use_focal_loss, use_cbfl, use_mixup, threshold = (
        VARS_FOR_EXPERIMENT  # controls the model configuration -> whether dropout/bn/augmentations are used etc
    )

    # make the parent folder all of the logs, images, model will go into
    if MODEL_NAME == "NA":
        print(f"Set a model name with the --name flag for training")
        exit(1)

    setup_folders()

    print(f"Evaluating Model {MODEL_NAME}")
    print(f"USE DROPOUT: {use_dropout}")
    print(f"CLASS WEIGHTS USED {use_weights}")
    print(f"BATCH NORM USED:  {use_batch_norm}")
    print(f"Focal Loss USED: {use_focal_loss}")
    print(f"START TIME {datetime.now()}")

    subsetter = DataSubsetter(
        ORIGINAL_DATASET_PATH,
        TRAIN_SET_PATH,
        VALIDATION_SET_PATH,
        TEST_SET_PATH,
        LABELS,
    )

    # check if the train.csv from which the train/val subset is created exists - if not prompt to download
    if not (os.path.exists(ORIGINAL_DATASET_PATH)):
        print(
            "Original Dataset not found \n Please download train.csv from: https://www.kaggle.com/datasets/ashery/chexpert"
        )
        exit(1)
    # check if the folder of images that will be used in the dataloaders exists - if not prompt to download
    if not (os.path.exists(IMAGES_PATH)):
        print(
            "ERROR: Images not dowloaded \n please download the train folder from: https://www.kaggle.com/datasets/ashery/chexpert"
        )
        exit(1)

    if not (os.path.exists(TRAIN_SET_PATH) and os.path.exists(VALIDATION_SET_PATH)):
        print("subsetting data to create train and validation files")
        subsetter.create_subset_train_validation()
        # if the train/val split needs to be remade
        # remake the test set also to ensure test images arent in any other set
        subsetter.create_test_data_csv()
    else:
        print("Train / Valid Subsets already created. Loader prep initializing..")

    # if only the test set is gone just remake the tes set
    if not os.path.exists(TEST_SET_PATH):
        print("creating test dataset now")
        subsetter.create_test_data_csv()

    subsetter.calculate_patient_overlap()

    dataloader_for_training, data_loader_for_validation = prepare_data(
        augment_transforms, use_clahe, IMAGES_PATH, TRAIN_SET_PATH, VALIDATION_SET_PATH
    )
    data_loader_for_testing = prepare_test_data(TEST_SET_PATH)

    class_weights = calculate_class_weights(TRAIN_SET_PATH) if use_weights else None

    if use_focal_loss:
        alpha = calculate_normalized_inverse_frequency_focal_loss(TRAIN_SET_PATH)
        print(f"alpha shape : {alpha.size()}")
        gamma = 2 # as recommended by the paper
        loss_fn = FocalLoss(alpha, gamma)
    elif use_cbfl:
        beta = calculate_class_freq_cbfl(TRAIN_SET_PATH)
        print(f"beta shape: {beta.size()}")
        gamma = 0.5
        loss_fn = ClassBalancedFocalLoss(beta=beta, gamma=gamma)
    else:
        loss_fn= BCEWithLogitsLoss(pos_weight=class_weights)

    if not os.path.exists(f"{MODEL_ROOT}/{MODEL_NAME}.pt"):
        model = BaselineModel(use_dropout=use_dropout, use_batch_norm=use_batch_norm)
        optimizer = Adam(model.parameters(), lr=1e-4)

        print("no previous models, training now")
        trainer = Trainer(
            model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=dataloader_for_training,
            validation_loader=data_loader_for_validation,
            class_weights=class_weights,
            NUM_EPOCHS=NUM_EPOCHS,
            use_mixup=use_mixup,
            threshold=threshold
        )
        train_losses, validation_losses, epochs = trainer.train_model()
        plot_loss(train_losses, validation_losses, epochs)
    else:
        print("previous models found!")
        model = torch.load(MODEL_ROOT / f"{MODEL_NAME}.pt", weights_only=False)

    post_train_model = model
    print(post_train_model)

    print("EVALUATION STARTING")
    eval_loop = EvaluationLoop(data_loader_for_testing, post_train_model, loss_fn)
    # get the per-class thresholds for the model
    if threshold == "optimal":
        with open(MODEL_ROOT / "thresholds.json", 'r', encoding="utf-8") as threshold_file:
            data = json.load(threshold_file)
            threshold = data["thresholds"]
    else:
        threshold = [0.3]*13
    eval_loop.evaluate_model(threshold)

    generate_comparisons("results")
    print(f"Finished Evaluating {MODEL_NAME}")
    print(f"FINISH TIME {datetime.now()}")
