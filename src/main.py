import os
from datetime import datetime
from pathlib import Path

import torch

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from RunPathHolder import RunPathHolder, setup_folders
from constants.paths import (
    ORIGINAL_DATASET_PATH,
    IMAGES_PATH,
    TEST_SET_PATH,
    TRAIN_SET_PATH,
    VALIDATION_SET_PATH,
    CSV_PATHS,
    # setup_folders,
)

from constants.control_variables import LABELS, NUM_EPOCHS
from data_preparation import DataSubsetter, prepare_data, prepare_test_data
from evaluate import EvaluationLoop
from experiment_utils.arg_parser import parse_arguments
from model import BaselineModel
from trainer import Trainer
from utils import (
    calculate_class_weights,
    calculate_normalized_inverse_frequency_focal_loss,
    calculate_class_freq_cbfl,
)

from comparison_generator import generate_comparisons
from custom_loss_fns.focal_loss import FocalLoss
from custom_loss_fns.class_balanced_focal_loss import ClassBalancedFocalLoss
import json
import numpy as np


def make_reproducible():
    # setting random seeds for reproducibility
    np.random.seed(23)
    torch.manual_seed(23)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def check_data_exists(
    ORIGINAL_DATASET_PATH,
    TRAIN_SET_PATH,
    VALIDATION_SET_PATH,
    TEST_SET_PATH,
    LABELS,
    CSV_PATHS,
):

    subsetter = DataSubsetter(
        ORIGINAL_DATASET_PATH,
        TRAIN_SET_PATH,
        VALIDATION_SET_PATH,
        TEST_SET_PATH,
        LABELS,
        CSV_PATHS,
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

    subsetter.calculate_natural_label_coocurrance()
    subsetter.calculate_patient_overlap()
    subsetter.plot_imbalance()


def get_loss_function(use_weights, focal_loss_gamma, use_cbfl):
    class_weights = calculate_class_weights(TRAIN_SET_PATH) if use_weights else None
    # cant move if it is none -> baseline
    if class_weights is not None:
        class_weights = class_weights.to(device)

    if focal_loss_gamma is not None:
        alpha = calculate_normalized_inverse_frequency_focal_loss(TRAIN_SET_PATH)
        print(f"alpha shape : {alpha.size()}")
        alpha = alpha.to(device)
        gamma = focal_loss_gamma  # as recommended by the paper
        loss_fn = FocalLoss(alpha, gamma)
    elif use_cbfl:
        beta = calculate_class_freq_cbfl(TRAIN_SET_PATH)
        beta = beta.to(device)
        print(f"beta shape: {beta.size()}")
        gamma = 0.5
        loss_fn = ClassBalancedFocalLoss(beta=beta, gamma=gamma)
    else:
        loss_fn = BCEWithLogitsLoss(pos_weight=class_weights)

    return loss_fn


if __name__ == "__main__":
    make_reproducible()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EXPERIMENT_ROOT, MODEL_NAME, *VARS_FOR_EXPERIMENT = parse_arguments()

    (
        augment_transforms,
        use_weights,
        use_clahe,
        use_dropout,
        use_batch_norm,
        focal_loss_gamma,
        use_cbfl,
        use_mixup,
        threshold,
    ) = VARS_FOR_EXPERIMENT  # controls the model configuration -> whether dropout/bn/augmentations are used etc

    run_specific_paths = RunPathHolder(
        experiment_name=EXPERIMENT_ROOT, model_name=MODEL_NAME
    )

    setup_folders(run_path_holder=run_specific_paths, csv_paths=CSV_PATHS)

    check_data_exists(
        ORIGINAL_DATASET_PATH,
        TRAIN_SET_PATH,
        VALIDATION_SET_PATH,
        TEST_SET_PATH,
        LABELS,
        CSV_PATHS,
    )

    # make the parent folder all of the logs, images, model will go into
    if MODEL_NAME == "NA":
        print(f"Set a model name with the --name flag for training")
        exit(1)

    print(f"Evaluating Model {MODEL_NAME}")
    print(f"USE DROPOUT: {use_dropout}")
    print(f"CLASS WEIGHTS USED {use_weights}")
    print(f"BATCH NORM USED:  {use_batch_norm}")
    print(f"Focal Loss GAMMA if Used: {focal_loss_gamma}")
    print(f"START TIME {datetime.now()}")

    dataloader_for_training, data_loader_for_validation = prepare_data(
        augment_transforms,
        use_clahe,
        IMAGES_PATH,
        TRAIN_SET_PATH,
        VALIDATION_SET_PATH,
        run_specific_paths.model_root,
    )
    data_loader_for_testing = prepare_test_data(TEST_SET_PATH)

    loss_fn = get_loss_function(
        use_weights,
        focal_loss_gamma,
        use_cbfl,
    )
    model_path = run_specific_paths.model_root / f"{MODEL_NAME}.pt"
    if not os.path.exists(model_path):
        model = BaselineModel(use_dropout=use_dropout, use_batch_norm=use_batch_norm)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=1e-4)

        print("no previous models, training now")
        trainer = Trainer(
            path_holder=run_specific_paths,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=dataloader_for_training,
            validation_loader=data_loader_for_validation,
            NUM_EPOCHS=NUM_EPOCHS,
            use_mixup=use_mixup,
            threshold=threshold,
        )
        trainer.train_model()

    else:
        print("previous models found!")
        model = torch.load(model_path, weights_only=False, map_location=device)

    model.to(device)
    print(model)

    print("EVALUATION STARTING")
    eval_loop = EvaluationLoop(
        run_specific_paths, data_loader_for_testing, model, loss_fn
    )
    # get the per-class thresholds for the model
    # if the file exists, use optimal thersholds, otherwise don't
    # guards against using the --fixed flag by accident
    threshold_filepath = run_specific_paths.model_root / "thresholds.json"
    if threshold == "optimal":
        if os.path.exists(threshold_filepath):
            with open(threshold_filepath, "r", encoding="utf-8") as threshold_file:
                data = json.load(threshold_file)
                threshold = data["thresholds"]
        else:
            threshold = [0.3] * 13
    else:
        threshold = [0.3] * 13
    eval_loop.evaluate_model(threshold)

    print(f"Finished Evaluating {MODEL_NAME}")
    print(f"FINISH TIME {datetime.now()}")
