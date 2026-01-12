from torch import tensor, float32
from math import sqrt
import torch
import matplotlib.pyplot as plt

import pandas as pd
from constants import LABELS, MODEL_NAME
from pathlib import Path

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
        sum_pixel_vals += torch.mean(train_features) # calculate the mean for every image in the batch - uses every value in the batch
        sum_pixel_vals_squared += torch.mean(train_features ** 2)

    mean = sum_pixel_vals / batches_amount # E[X] on wikipedia
    variance = (sum_pixel_vals_squared / batches_amount - mean ** 2) # E[X^2] on wikipedia
    standard_deviation = sqrt(variance)

    #print(mean, standard_deviation)

    return (mean, standard_deviation)

def plot_loss(training_losses,validation_losses, epochs):
    
       plt.plot(epochs, training_losses, label="Training Loss")
       plt.plot(epochs, validation_losses, label="Validation Loss")
       plt.ylabel("Average Loss / epoch")
       plt.xlabel("Number of epochs completed")
       plt.legend()
       plt.title("Training and Validation losses over epochs")
       plt.savefig(f'{MODEL_NAME}/{MODEL_NAME} loss graph')

       return

def calculate_class_weights(train_file):
    class_weights = []
    train_csv = pd.read_csv(train_file)
    for LABEL in LABELS:
        num_positives = train_csv[LABEL].sum()
        # weight = |negative samples| / |positive samples|
        num_negatives = train_csv.shape[0] - num_positives
        label_class_weight = num_negatives / num_positives

        class_weights.append(label_class_weight)
    print(class_weights)

    return 


def save_model(model):
    torch.save(model.state_dict(), f"{MODEL_NAME}/{MODEL_NAME}.pt")
    return

def write_train_loss_to_file(epoch_list, loss_to_plot):
    # add the losses to a file as logs
    loss_file = Path(f"{MODEL_NAME}/train_data/avg_epoch_loss.txt")
    loss_file.parent.mkdir(exist_ok=True, parents=True)
    with open(loss_file, "w") as file:
        for e, l in zip(epoch_list, loss_to_plot):
            file.write(f"Epoch {e} loss: {l}\n")

    return
