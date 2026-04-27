from torch import tensor, float32
from math import sqrt
import torch
import matplotlib.pyplot as plt

import pandas as pd
from constants.control_variables import LABELS
from pathlib import Path
import csv


def calculate_mean_and_standard_deviation(dataloader, device):
    """
    get the mean pixel value, and standard deviation for the entire dataset

    References:
        -  https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c/
        - https://en.wikipedia.org/wiki/Standard_deviation to understand more where the formula used comes from


    Arguments: dataloader: the dataloader which contains the dataset

    Returns: (mean: float, standard_deviation: float)
    """
    print("Starting to Calculate Mean and Standard Deviation...")
    sum_pixel_vals = 0
    sum_pixel_vals_squared = 0
    batches_amount = len(dataloader)

    # As per wikipedia X is some variable - for me it is the pixel value
    # for each batch of the dataset, calculate the mean pixel value for each batch
    # and divide by the number of batches to get the overall mean for the dataset E[X]
    # as per wikipedia
    # for standard deviation:
    # square each pixel value in the batch, and find the mean for this squared batch
    # E[X^2] as per wikipedia
    # find the variance -> E[X^2] - E[X]^2 as per wikipedia
    # this is the variance -> square root to get the standard deviation
    for train_features, train_labels in dataloader:
        # print(train_features.size()) # a batch of 25, 224x224 images [batch size, channels (1 because grayscale), height, width
        train_features = train_features.to(device).to(dtype=float32)
        sum_pixel_vals += torch.mean(
            train_features
        )  # calculate the mean for every image in the batch - uses every value in the batch
        sum_pixel_vals_squared += torch.mean(train_features**2)

    mean = sum_pixel_vals / batches_amount  # E[X] on wikipedia
    variance = sum_pixel_vals_squared / batches_amount - mean**2  # E[X^2] on wikipedia
    standard_deviation = sqrt(variance)

    print(
        f"Finished Calculating Mean and Standard Deviation: Mean: {mean}, Standard Deviation: {standard_deviation}"
    )

    return (float(mean), float(standard_deviation))


def plot_loss_ap(training_losses, validation_ap, epochs, save_path):
    """Plots the train and validatoin losses to a graph and saves them

    Args:
        training_losses (List): list of train losses
        validation_ap (list): list of validation losses
        epochs (list): list of epoch numbers
    """

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 10))
    ax[0].plot(epochs, training_losses, label="Training Loss")
    ax[0].set_title("Training loss / epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Epoch")

    ax[1].plot(
        epochs, validation_ap, label="Validation Average Precision", color="orange"
    )
    ax[1].set_title("Validation Average Precision / epoch")
    ax[1].set_ylabel("Average Precision")
    ax[1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    plt.close(fig)

    return


def calculate_class_weights(train_file):
    """Calculates the class weights using the ratio of negative:positive examples for a given class

    Args:
        train_file (str): the path of the train csv

    Returns:
        Tensor: a tensor containing the weight values for each class
    """
    class_weights = []
    train_csv = pd.read_csv(train_file)
    for LABEL in LABELS:
        num_positives = train_csv[LABEL].sum()
        # weight = |negative samples| / |positive samples|
        num_negatives = train_csv.shape[0] - num_positives
        label_class_weight = num_negatives / num_positives

        class_weights.append(label_class_weight)

    return torch.tensor(class_weights)


def save_model(model, save_path):
    """Saves a model to the model path

    Args:
        model (BaselineModel): the model to be saved
        save_path: the path to save the model to
    """
    torch.save(model, save_path)
    return


def write_train_loss_to_file(epoch_list, loss_to_plot, validation_ap, loss_file_path):
    """Write the train losses to a file

    Args:
        epoch_list (List): list of epoch numbers
        loss_to_plot (List): list of train losses
        validation_ap (list): list of validation losses
        loss_file_path: the path to save the losses and AP to
    """
    loss_headings = ["Epoch", "Train Loss", "Validation Average Precision"]
    with open(loss_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(loss_headings)
        for epoch, train_loss, val_ap in zip(epoch_list, loss_to_plot, validation_ap):
            writer.writerow([epoch, train_loss, val_ap])

    return


def calculate_class_freq_cbfl(TRAIN_SET_PATH):
    """Calculate the class weights for Class Balanced Focal Loss based on the Effective Number of Samples

    Args:
        TRAIN_SET_PATH (Path): The path of the train set to use to get frequencies

    Returns:
        Tensor: a tensor containing the class weights in the form[[pos_i, neg_i]] for each class i
    """

    class_frequencies = []
    df = pd.read_csv(TRAIN_SET_PATH)

    for label in LABELS:
        n_pos = df[label].sum()
        n_neg = df.shape[0] - n_pos

        class_frequencies.append([n_pos, n_neg])

    # (1-beta) / (1-beta^n_y)

    beta = 0.99
    beta_class_weights = []
    for pos_neg_pair in class_frequencies:
        pos_weight = (1 - beta) / (1 - beta ** pos_neg_pair[0])
        neg_weight = (1 - beta) / (1 - beta ** pos_neg_pair[1])
        beta_class_weights.append([pos_weight, neg_weight])

    # normalize so that each pair of weights sum to 2 - in sigmoid multi-label becomes 13 binary choices
    # x_i' = x_i(N/Sum(x_n))
    print(f"beta_class frequencies {beta_class_weights}")
    normalized_beta_class_weights = []
    for pair in beta_class_weights:
        pos_weight, neg_weight = pair[0], pair[1]
        normalized_pos_weight = pos_weight * (2 / (sum([pos_weight, neg_weight])))
        normalized_neg_weight = neg_weight * (2 / (sum([pos_weight, neg_weight])))
        normalized_beta_class_weights.append(
            [normalized_pos_weight, normalized_neg_weight]
        )

    return torch.tensor(normalized_beta_class_weights)
