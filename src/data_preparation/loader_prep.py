from constants.paths import (
    IMAGES_PATH,
    MODEL_ROOT,
    MEAN_STD_PATH
)
import json
import os


import torch
from chexpert_dataset import ChexpertDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)  # to resize all images

from utils import calculate_mean_and_standard_deviation

def prepare_data(
    augment_transforms, use_clahe, IMAGES_PATH, TRAIN_SET_PATH, VALIDATION_SET_PATH
):
    """Prepare the train and validatoin dataloaders for use in training

    Args:
        augment_transforms (list): the list of augmentations to be applied to images in the train dataloader
        use_clahe (boolean): If true, clahe transformations will be applied
        IMAGES_PATH (string): file path to the images
        TRAIN_SET_PATH (string):path to train.csv -> the csv of images in the train set
        VALIDATION_SET_PATH (string): path to validation.csv -> the csv of images in the validation set

    Returns:
        tuple(DataLoader): Returns the train, and validation dataloaders
    """
    resize_transform = Resize(
        (224, 224)
    )  # some images are different sizes so resize them all to the same size
    train_dataset = ChexpertDataset(
        TRAIN_SET_PATH, IMAGES_PATH, transform=resize_transform
    )
    validation_dataset = ChexpertDataset(
        VALIDATION_SET_PATH, IMAGES_PATH, transform=resize_transform
    )

    print(f"dataset size: {train_dataset.__len__()}")

    # saving one image before and after transforms
    before_transform_img, _ = train_dataset[0]
    before_transform_img.save(f"{MODEL_ROOT}/before.png")

    dataset_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)

    calc_transforms = Compose([resize_transform, ToTensor()])
    calc_train_dataset = ChexpertDataset(
        TRAIN_SET_PATH, IMAGES_PATH, transform=calc_transforms
    )
    calc_loader = DataLoader(calc_train_dataset, batch_size=25, shuffle=True)

    if os.path.exists(MEAN_STD_PATH):
        with open(MEAN_STD_PATH, "r") as f:
            mean_std_data = json.load(f)
        mean = mean_std_data["mean"]
        standard_deviation = mean_std_data["std"]

    else:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mean, standard_deviation = calculate_mean_and_standard_deviation(calc_loader, device)
        mean_std_dict = {"mean": mean, "std": standard_deviation}

        with open(MEAN_STD_PATH, 'w') as norm_const_file:
            json.dump(mean_std_dict, norm_const_file)
        print(f"Calcd mean and std: {mean, standard_deviation}")

    print(f"mean = {mean}, standard deviation = {standard_deviation}")

    common_transforms = [
        resize_transform,
        ToTensor(),
        Normalize(mean=mean, std=standard_deviation),
    ]

    # Resize -> [Augments] -> ToTensor and Normalize
    train_transforms = (
        [common_transforms[0]] + augment_transforms + common_transforms[1:]
    )

    transforms = Compose(common_transforms)
    train_transforms = Compose(train_transforms)

    print(f"train transforms{train_transforms}")
    print(f"valid transforms{transforms}")

    after_normalization_train_dataset = ChexpertDataset(
        TRAIN_SET_PATH, IMAGES_PATH, transform=train_transforms, use_clahe=use_clahe
    )
    after_normalization_train_loader = DataLoader(
        after_normalization_train_dataset, batch_size=25, shuffle=True
    )

    after_normalization_validation_dataset = ChexpertDataset(
        VALIDATION_SET_PATH, IMAGES_PATH, transform=transforms
    )
    after_normalization_validation_loader = DataLoader(
        after_normalization_validation_dataset, batch_size=25, shuffle=False
    )

    # checking the after normalization dataset
    image, labels = after_normalization_train_dataset[0]
    print(f"image shape: {image.shape}")  # grayscale(1 channel) 224x224 image

    # check the range of values after normalizing
    print(f"max pixel value: {image.max()}, min pixel value: {image.min()}")

    print(f"labels: {labels.shape}")  # checking the labels exist properly for the image

    # checking the data is loaded - from PyTorch DataLoader Documentation and save an image
    # after transforms are applied
    train_features, train_labels = next(iter(after_normalization_train_loader))
    save_image(train_features[0], f"{MODEL_ROOT}/after.png")
    print(f"Feature shape: {train_features.size()} ")
    print(f"label batch shape: {train_labels.size()}")

    return after_normalization_train_loader, after_normalization_validation_loader


def prepare_test_data(TEST_SET_PATH):
    """Prepares the test dataloader to be used for evaluating the models

    Args:
        TEST_SET_PATH (string): path to test.csv -> the csv of images and their labels in the test split

    Returns:
        DataLoader: the test dataloader
    """
    resize_transform = Resize(
        (224, 224)
    )  # some images are different sizes so resize them all to the same size
    test_dataset = ChexpertDataset(
        TEST_SET_PATH, IMAGES_PATH, transform=resize_transform
    )

    test_dataset_loader = DataLoader(test_dataset, batch_size=25, shuffle=False)

    with open(MEAN_STD_PATH, "r") as f:
        mean_std_data = json.load(f)
    mean = mean_std_data["mean"]
    standard_deviation = mean_std_data["std"]
    print(f"Test Mean: {mean}, Test STD: {standard_deviation}")

    transforms = Compose(
        [resize_transform, ToTensor(), Normalize(mean=mean, std=standard_deviation)]
    )

    after_normalization_test_dataset = ChexpertDataset(
        TEST_SET_PATH, IMAGES_PATH, transform=transforms
    )
    after_normalization_test_loader = DataLoader(
        after_normalization_test_dataset, batch_size=25, shuffle=False
    )

    return after_normalization_test_loader
