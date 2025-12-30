import torch

from chexpert_dataset import ChexpertDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,Normalize, Resize # to resize all images
from utils import calculate_mean_and_standard_deviation

from model import BaselineModel 
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from constants import NUM_EPOCHS

import matplotlib.pyplot as plt
from utils import plot_training_loss

def prepare_data():

    resize_transform = Resize((224, 224)) # some images are different sizes so resize them all to the same size
    dataset = ChexpertDataset("subset.csv", "D:/dataset fyp/", transform=resize_transform)


    print(f"dataset size: {dataset.__len__()}")

    # testing with just 1 image for now to test logic
    #image, labels = dataset[0]
    #print(f"label type: {type(labels)}")

    #print(f"image shape: {image.shape}\n image : {image}\n") # grayscale(1 channel) 224x224 image
    #print(f"labels: {labels.shape} || {labels}\n") # checking the labels exist properly for the image

    dataset_loader = DataLoader(dataset, batch_size=25, shuffle=True)



    ####### TEMP: Temporarily hard code mean and std during testing
    #mean, standard_deviation = calculate_mean_and_standard_deviation(dataset_loader)
    mean = 0.5062767267227173
    standard_deviation = 0.28674584034677253



    print(f"mean = {mean}, standard deviation = {standard_deviation}")

    transforms = Compose([
        resize_transform,
        Normalize(mean=mean, std=standard_deviation)
    ])

    after_normalization_dataset = ChexpertDataset("subset.csv", "D:/dataset fyp/", transform=transforms)
    after_normalization_loader = DataLoader(after_normalization_dataset, batch_size=25, shuffle=True) 

    # checking the after normalization dataset
    image, labels = after_normalization_dataset[0]
    print(f"label type: {type(labels)}")
    print(f"image shape: {image.shape}\n image : {image}\n") # grayscale(1 channel) 224x224 image

    # check the range of values after normalizing
    print(f"max pixel value: {image.max()}, min pixel value: {image.min()}")

    print(f"labels: {labels.shape} || {labels}\n") # checking the labels exist properly for the image

    # # checking the data is loaded - from PyTorch DataLoader Documentation
    train_features, train_labels = next(iter(after_normalization_loader))
    print(f"Feature shape: {train_features.size()} ")
    print(f"label batch shape: {train_labels.size()}")

    return after_normalization_loader


def train_model(model, dataloader):
    # right now just 1 epoch as a check that it works

    baseline_train = model
    loss_function = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters())

    loss_to_plot = [] # list of loss values to plot
    epoch_list = [] # corresponding epoch of each loss value during training

    loss_so_far = 0


    print(baseline_train)

    for epoch in range(NUM_EPOCHS):

        current_batch_num = 0
        total_loss_for_epoch = 0
        model.train()
        for i, data in enumerate(dataloader):
            images, labels = data
            optimizer.zero_grad()
            outputs = baseline_train(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss_for_epoch += loss.item()
            current_batch_num += 1

            if i % 50  == 49: # print the loss every 10 batches
                loss_so_far = total_loss_for_epoch / current_batch_num # avg loss up to current batch number
                print(f"loss so far at batch {current_batch_num}: {loss_so_far}")

        # after each epoch during training, add the epoch number and loss value
        # to the list to be visualized

        epoch_average_loss =  total_loss_for_epoch/(current_batch_num) # average loss total so far up to the current epoch
        epoch_list.append(epoch+1)
        loss_to_plot.append(epoch_average_loss)


        print(f"Epoch {epoch+1} complete. Final Avg Loss for this epoch: {epoch_average_loss}") # average loss across all batches

    print(f"training completed")
    return (model, loss_to_plot, epoch_list)
       

if __name__ == "__main__":
    dataloader_for_training = prepare_data()
    model = BaselineModel()

    print("###############")
    post_train_model, losses, epochs = train_model(model, dataloader_for_training)
    torch.save(post_train_model.state_dict(), "baseline_test.pt")
    plot_training_loss(losses, epochs)

    

