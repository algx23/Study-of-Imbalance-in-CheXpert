from torch import tensor, float32
from math import sqrt
import torch
import matplotlib.pyplot as plt

import pandas as pd
from constants.control_variables import LABELS, MODEL_NAME
from constants.paths import MODEL_ROOT, TRAIN_DATA_PATH
from pathlib import Path
import csv


def calculate_mean_and_standard_deviation(dataloader):
    """
    get the mean pixel value, and standard deviation for the entire dataset

    References:
        -  https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c/
        - https://en.wikipedia.org/wiki/Standard_deviation to understand more where the formula used comes from


    Arguments: dataloader: the dataloader which contains the dataset

    Returns: (mean: float, standard_deviation: float)
    """
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
        train_features = train_features.to(dtype=float32)
        sum_pixel_vals += torch.mean(
            train_features
        )  # calculate the mean for every image in the batch - uses every value in the batch
        sum_pixel_vals_squared += torch.mean(train_features**2)

    mean = sum_pixel_vals / batches_amount  # E[X] on wikipedia
    variance = sum_pixel_vals_squared / batches_amount - mean**2  # E[X^2] on wikipedia
    standard_deviation = sqrt(variance)

    # print(mean, standard_deviation)

    return (mean, standard_deviation)


def plot_loss(training_losses, validation_losses, epochs):
    """Plots the train and validatoin losses to a graph and saves them

    Args:
        training_losses (List): list of train losses
        validation_losses (list): list of validation losses
        epochs (list): list of epoch numbers
    """

    plt.plot(epochs, training_losses, label="Training Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")
    plt.ylabel("Average Loss / epoch")
    plt.xlabel("Number of epochs completed")
    plt.legend()
    plt.title("Training and Validation losses over epochs")
    plt.savefig(f"{MODEL_ROOT}/{MODEL_NAME}_loss_graph.png")
    plt.clf()

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


def save_model(model):
    """Saves a model to the model path

    Args:
        model (BaselineModel): the model to be saved
    """
    torch.save(model, f"{MODEL_ROOT}/{MODEL_NAME}.pt")
    return


def write_train_loss_to_file(epoch_list, loss_to_plot, validation_losses):
    """Write the train losses to a file

    Args:
        epoch_list (List): list of epoch numbers
        loss_to_plot (List): list of train losses
        validation_losses (list): list of validation losses
    """
    # add the losses to a file as logs
    loss_file = f"{TRAIN_DATA_PATH}/avg_epoch_loss.csv"

    loss_headings = ["Epoch", "Train Loss", "Validation Loss"]
    with open(loss_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(loss_headings)
        for epoch, train_loss, val_loss in zip(
            epoch_list, loss_to_plot, validation_losses
        ):
            writer.writerow([epoch, train_loss, val_loss])

    return
