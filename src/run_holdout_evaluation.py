import argparse
import json
import os

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import pandas as pd
import torch

from RunPathHolder import RunPathHolder, setup_folders
from constants.paths import (
    CSV_PATHS,
    TEST_SET_PATH,
    TRAIN_SET_PATH,
    VALIDATION_SET_PATH,
)
from data_preparation.loader_prep import prepare_test_data
from evaluate import EvaluationLoop
from experiment_utils.arg_parser import parse_arguments
from main import get_loss_function


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


def create_final_holdout():
    test = pd.read_csv("data/prepared_test.csv")
    subset = pd.read_csv("subset.csv")

    all_data = pd.read_csv("prepared data.csv")

    # adapted from: https://stackoverflow.com/questions/44706485/how-to-remove-rows-in-a-pandas-dataframe-if-the-same-row-exists-in-another-dataf
    train_val_subset_removed = (
        pd.merge(all_data, subset, indicator=True, how="outer")
        .query("_merge=='left_only'")
        .drop("_merge", axis=1)
    )
    data_without_test = (
        pd.merge(train_val_subset_removed, test, indicator=True, how="outer")
        .query("_merge=='left_only'")
        .drop("_merge", axis=1)
    )

    # split this to get the test set
    paths_to_images = data_without_test["Path"]
    labels_for_each_row = data_without_test[LABELS].values

    test_split = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=2000, random_state=42
    )
    for _, test_index in test_split.split(paths_to_images, labels_for_each_row):
        X_hold_out = paths_to_images.iloc[test_index]
        Y_hold_out = labels_for_each_row[test_index]

    hold_out_df = X_hold_out.to_frame().reset_index(drop=True)
    hold_out_df[LABELS] = Y_hold_out
    print("Hold out distribution: \n")
    print(hold_out_df[LABELS].mean() * 100)
    hold_out_df.to_csv(CSV_PATHS / "holdout.csv", index=False)

    train_set = pd.read_csv(TRAIN_SET_PATH)
    validation_set = pd.read_csv(VALIDATION_SET_PATH)
    test_set = pd.read_csv(TEST_SET_PATH)

    holdout_img_leak = len(
        (set(subset["Path"]) | set(test["Path"])) & set(hold_out_df["Path"])
    )
    train_patient_ids = set(train_set["Path"].str.split("/").str[2])
    val_patient_ids = set(validation_set["Path"].str.split("/").str[2])
    test_patient_ids = set(test_set["Path"].str.split("/").str[2])
    holdout_patient_ids = set(hold_out_df["Path"].str.split("/").str[2])

    holdout_leakage = len(
        holdout_patient_ids & (train_patient_ids | val_patient_ids | test_patient_ids)
    )
    print(f"IMAGE HOLD OUT LEAK: {holdout_img_leak}")
    print(f"Patient Leakage in HOLDOUT: {holdout_leakage}")

    return


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name, model_name, *_ = parse_arguments()

    HOLDOUT_PATH = CSV_PATHS / "holdout.csv"

    if not os.path.exists(HOLDOUT_PATH):
        create_final_holdout()

    final_eval_path_holder = RunPathHolder(
        experiment_name=experiment_name, model_name=model_name
    )
    setup_folders(final_eval_path_holder, csv_paths=CSV_PATHS)

    # hard-code using BCE Loss because in evaluation, the loss is purely for display purposes only
    loss_fn = get_loss_function(use_weights=True, focal_loss_gamma=2, use_cbfl=False)

    model = torch.load(
        final_eval_path_holder.model_root / f"{model_name}.pt",
        weights_only=False,
        map_location=device,
    )

    holdout_loader = prepare_test_data(HOLDOUT_PATH)
    final_eval = EvaluationLoop(final_eval_path_holder, holdout_loader, model, loss_fn)
    threshold_filepath = final_eval_path_holder.model_root / "thresholds.json"

    if os.path.exists(threshold_filepath):
        with open(threshold_filepath, "r", encoding="utf-8") as threshold_file:
            data = json.load(threshold_file)
            threshold = data["thresholds"]
    else:
        threshold = [0.3] * 13

    final_eval.evaluate_model(thresholds=threshold)
