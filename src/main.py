from chexpert_dataset import ChexpertDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,Normalize, Resize # to resize all images
from utils import calculate_mean_and_standard_deviation

<<<<<<< Updated upstream
resize_transform = Resize((224, 224)) # some images are different sizes so resize them all to the same size
dataset = ChexpertDataset("subset.csv", "D:/dataset fyp/", transform=resize_transform)


print(f"dataset size: {dataset.__len__()}")
=======
from model import BaselineModel 
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from constants import NUM_EPOCHS

import matplotlib.pyplot as plt
from utils import plot_training_loss

def prepare_data():

    resize_transform = Resize((224, 224)) # some images are different sizes so resize them all to the same size
    train_dataset = ChexpertDataset("train.csv", "D:/dataset fyp/", transform=resize_transform)
    validation_dataset = ChexpertDataset("validation.csv", "D:/dataset fyp/", transform=resize_transform)


    print(f"dataset size: {train_dataset.__len__()}")
>>>>>>> Stashed changes

# testing with just 1 image for now to test logic
#image, labels = dataset[0]
#print(f"label type: {type(labels)}")

#print(f"image shape: {image.shape}\n image : {image}\n") # grayscale(1 channel) 224x224 image
#print(f"labels: {labels.shape} || {labels}\n") # checking the labels exist properly for the image

<<<<<<< Updated upstream
dataset_loader = DataLoader(dataset, batch_size=25, shuffle=True)



####### TEMP: Temporarily hard code mean and std during testing
#mean, standard_deviation = calculate_mean_and_standard_deviation(dataset_loader)
mean = 0.5062767267227173
standard_deviation = 0.28674584034677253


=======
    dataset_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)



    ####### TEMP: Temporarily hard code mean and std during testing
    mean, standard_deviation = calculate_mean_and_standard_deviation(dataset_loader)
    #mean = 0.5062767267227173
    #standard_deviation = 0.28674584034677253
>>>>>>> Stashed changes

print(f"mean = {mean}, standard deviation = {standard_deviation}")

transforms = Compose([
    resize_transform,
    Normalize(mean=mean, std=standard_deviation)
])

<<<<<<< Updated upstream
after_normalization_dataset = ChexpertDataset("subset.csv", "D:/dataset fyp/", transform=transforms)
after_normalization_loader = DataLoader(after_normalization_dataset, batch_size=25, shuffle=True) 

# checking the after normalization dataset
image, labels = after_normalization_dataset[0]
print(f"label type: {type(labels)}")
print(f"image shape: {image.shape}\n image : {image}\n") # grayscale(1 channel) 224x224 image
=======
    after_normalization_train_dataset = ChexpertDataset("train.csv", "D:/dataset fyp/", transform=transforms)
    after_normalization_validation_dataset = ChexpertDataset("validation.csv", "D:/dataset fyp/", transform=transforms)
    after_normalization_loader = DataLoader(after_normalization_train_dataset, batch_size=25, shuffle=True) 

    # checking the after normalization dataset
    image, labels = after_normalization_train_dataset[0]
    print(f"label type: {type(labels)}")
    print(f"image shape: {image.shape}\n image : {image}\n") # grayscale(1 channel) 224x224 image
>>>>>>> Stashed changes

# check the range of values after normalizing
print(f"max pixel value: {image.max()}, min pixel value: {image.min()}")

print(f"labels: {labels.shape} || {labels}\n") # checking the labels exist properly for the image

# # checking the data is loaded - from PyTorch DataLoader Documentation
train_features, train_labels = next(iter(after_normalization_loader))
print(f"Feature shape: {train_features.size()} ")
print(f"label batch shape: {train_labels.size()}")
