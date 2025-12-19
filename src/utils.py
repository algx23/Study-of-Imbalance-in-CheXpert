from torch import tensor, float32
from math import sqrt
import torch


def calculate_mean_and_standard_deviation(dataloader):
    """
    get the mean pixel value, and standard deviation for the entire dataset

    References:
    - https://www.youtube.com/watch?time_continue=359&v=y6IEcEBRZks&embeds_referring_euri=https%3A%2F%2Fwww.google.com%2Fsearch%3Fsca_esv%3Ddb2aaf917c165eff%26udm%3D7%26q%3Dwork%2Bout%2Bstandard%2Bdeviation%2Bpytorch%26sa%3DX%26ved%3D2ahUKEwja0LajisqRAxW&source_ve_path=MzY4NDIsMjM4NTE
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
