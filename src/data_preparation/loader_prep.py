from constants.paths import (IMAGES_PATH,
                             TRAIN_SET_PATH,
                             VALIDATION_SET_PATH,
                             TEST_SET_PATH,
                             MODEL_ROOT
                       )


import torch
from chexpert_dataset import ChexpertDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import Compose,Normalize, Resize, ToTensor # to resize all images


def prepare_data(augment_transforms, use_clahe, IMAGES_PATH, TRAIN_SET_PATH, VALIDATION_SET_PATH):

    resize_transform = Resize((224, 224)) # some images are different sizes so resize them all to the same size
    train_dataset = ChexpertDataset(TRAIN_SET_PATH, IMAGES_PATH, transform=resize_transform)
    validation_dataset = ChexpertDataset(VALIDATION_SET_PATH, IMAGES_PATH, transform=resize_transform)


    print(f"dataset size: {train_dataset.__len__()}")

    # testing with just 1 image for now to test logic
    #image, labels = dataset[0]
    #print(f"label type: {type(labels)}")

    #print(f"image shape: {image.shape}\n image : {image}\n") # grayscale(1 channel) 224x224 image
    #print(f"labels: {labels.shape} || {labels}\n") # checking the labels exist properly for the image


    # saving one image before and after transforms
    before_transform_img, _ = train_dataset[0]
    before_transform_img.save(f"{MODEL_ROOT}/before.png")

    dataset_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)

    ####### TEMP: Temporarily hard code mean and std during testing
    #mean, standard_deviation = calculate_mean_and_standard_deviation(dataset_loader)
    mean =0.5062857270240784
    standard_deviation = 0.2867498937006307

    print(f"mean = {mean}, standard deviation = {standard_deviation}")

    common_transforms = [
        resize_transform,
        ToTensor(),
        Normalize(mean=mean, std=standard_deviation)
    ]

    # Resize -> [Augments] -> ToTensor and Normalize
    train_transforms = [common_transforms[0]] + augment_transforms + common_transforms[1:]

    transforms = Compose(common_transforms)
    train_transforms = Compose(train_transforms)

    print(f"train transforms{train_transforms}")
    print(f"valid transforms{transforms}")

    after_normalization_train_dataset = ChexpertDataset(TRAIN_SET_PATH, IMAGES_PATH, transform=train_transforms, use_clahe=use_clahe)
    after_normalization_train_loader = DataLoader(after_normalization_train_dataset, batch_size=25, shuffle=True)

    after_normalization_validation_dataset = ChexpertDataset(VALIDATION_SET_PATH, IMAGES_PATH, transform=transforms)
    after_normalization_validation_loader = DataLoader(after_normalization_validation_dataset, batch_size=25, shuffle=True)

    # checking the after normalization dataset
    image, labels = after_normalization_train_dataset[0]
    print(f"label type: {type(labels)}")
    print(f"image shape: {image.shape}\n image : {image}\n") # grayscale(1 channel) 224x224 image

    # check the range of values after normalizing
    print(f"max pixel value: {image.max()}, min pixel value: {image.min()}")

    print(f"labels: {labels.shape} || {labels}\n") # checking the labels exist properly for the image

    # # checking the data is loaded - from PyTorch DataLoader Documentation
    train_features, train_labels = next(iter(after_normalization_train_loader))
    save_image(train_features[0], f"{MODEL_ROOT}/after.png")
    print(f"Feature shape: {train_features.size()} ")
    print(f"label batch shape: {train_labels.size()}")

    return after_normalization_train_loader, after_normalization_validation_loader


def prepare_test_data(TEST_SET_PATH):
    resize_transform = Resize((224, 224)) # some images are different sizes so resize them all to the same size
    test_dataset = ChexpertDataset(TEST_SET_PATH, IMAGES_PATH, transform=resize_transform)

    test_dataset_loader = DataLoader(test_dataset, batch_size=25, shuffle=True)

    mean = 0.5062857270240784
    standard_deviation = 0.2867498937006307
    #mean, standard_deviation = calculate_mean_and_standard_deviation(test_dataset_loader)
    print(f"Test Mean: {mean}, Test STD: {standard_deviation}")


    transforms = Compose([
        resize_transform,
        ToTensor(),
        Normalize(mean=mean, std=standard_deviation)
    ])

    after_normalization_test_dataset = ChexpertDataset(TEST_SET_PATH, IMAGES_PATH, transform=transforms)
    after_normalization_test_loader = DataLoader(after_normalization_test_dataset, batch_size=25, shuffle=True)

    return after_normalization_test_loader
