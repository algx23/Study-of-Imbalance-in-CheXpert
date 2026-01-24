import os
import pandas as pd
from datetime import datetime
from experiment_utils.arg_parser import parse_arguments

from sample_dataset_to_5000 import (create_subset_train_validation,
                                    create_test_data_csv)

import torch
from chexpert_dataset import ChexpertDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,Normalize, Resize # to resize all images
from utils import (calculate_mean_and_standard_deviation,
                   calculate_class_weights,
                   save_model,
                   write_train_loss_to_file)


from model import BaselineModel
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from constants import NUM_EPOCHS, LABELS, MODEL_NAME
from validation_loss_checker import ValidationLossChecker

import matplotlib.pyplot as plt
from utils import plot_loss

from pathlib import Path

from sklearn.metrics import (multilabel_confusion_matrix,
                             classification_report,
                             roc_auc_score,
                             ConfusionMatrixDisplay)
import numpy as np

# augmentation imports
from torchvision.transforms import (RandomRotation,
                                    RandomHorizontalFlip)


def prepare_data(augment_transforms):

    resize_transform = Resize((224, 224)) # some images are different sizes so resize them all to the same size
    train_dataset = ChexpertDataset("train.csv", "D:/dataset fyp/", transform=resize_transform)
    validation_dataset = ChexpertDataset("validation.csv", "D:/dataset fyp/", transform=resize_transform)


    print(f"dataset size: {train_dataset.__len__()}")

    # testing with just 1 image for now to test logic
    #image, labels = dataset[0]
    #print(f"label type: {type(labels)}")

    #print(f"image shape: {image.shape}\n image : {image}\n") # grayscale(1 channel) 224x224 image
    #print(f"labels: {labels.shape} || {labels}\n") # checking the labels exist properly for the image

    dataset_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)



    ####### TEMP: Temporarily hard code mean and std during testing
    #mean, standard_deviation = calculate_mean_and_standard_deviation(dataset_loader)
    mean =0.5062857270240784
    standard_deviation = 0.2867498937006307

    print(f"mean = {mean}, standard deviation = {standard_deviation}")

    common_transforms = [
        resize_transform,
        Normalize(mean=mean, std=standard_deviation)
    ]
    train_transforms = common_transforms + augment_transforms

    transforms = Compose(common_transforms)
    train_transforms = Compose(train_transforms)

    after_normalization_train_dataset = ChexpertDataset("train.csv", "D:/dataset fyp/", transform=train_transforms)
    after_normalization_train_loader = DataLoader(after_normalization_train_dataset, batch_size=25, shuffle=True)
    after_normalization_validation_dataset = ChexpertDataset("validation.csv", "D:/dataset fyp/", transform=transforms)
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
    print(f"Feature shape: {train_features.size()} ")
    print(f"label batch shape: {train_labels.size()}")


    return after_normalization_train_loader, after_normalization_validation_loader

def prepare_test_data():
    resize_transform = Resize((224, 224)) # some images are different sizes so resize them all to the same size
    test_dataset = ChexpertDataset("prepared_test.csv", "D:/dataset fyp/", transform=resize_transform)

    test_dataset_loader = DataLoader(test_dataset, batch_size=25, shuffle=True)

    mean = 0.5062857270240784
    standard_deviation = 0.2867498937006307
    #mean, standard_deviation = calculate_mean_and_standard_deviation(test_dataset_loader)
    print(f"Test Mean: {mean}, Test STD: {standard_deviation}")


    transforms = Compose([
        resize_transform,
        Normalize(mean=mean, std=standard_deviation)
    ])

    after_normalization_test_dataset = ChexpertDataset("prepared_test.csv", "D:/dataset fyp/", transform=transforms)
    after_normalization_test_loader = DataLoader(after_normalization_test_dataset, batch_size=25, shuffle=True)

    return after_normalization_test_loader


def train_model(model, train_dataloader, validation_dataloader, class_weights=None):

    validation_loss_checker = ValidationLossChecker(
        min_improvement=0.0001,
        epochs_to_wait=10,
        validation_loader = validation_dataloader
    )
    loss_function = BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = Adam(model.parameters(), lr=0.0001)

    train_losses = [] # list of loss values to plot
    epoch_list = [] # corresponding epoch of each loss value during training

    loss_so_far = 0


    print(model)

    for epoch in range(NUM_EPOCHS):

        current_batch_num = 0
        total_loss_for_epoch = 0
        model.train()
        for i, data in enumerate(train_dataloader):
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss_for_epoch += loss.item()
            current_batch_num += 1

            if i % 50 == 49: # print the loss every 10 batches
                loss_so_far = total_loss_for_epoch / current_batch_num # avg loss up to current batch number
                print(f"loss so far at batch {current_batch_num}: {loss_so_far}")

        # after each epoch during training, add the epoch number and loss value
        # to the list to be visualized
        epoch_average_loss = total_loss_for_epoch/(current_batch_num) # average loss total so far up to the current epoch
        epoch_list.append(epoch+1)
        train_losses.append(epoch_average_loss)

        if validation_loss_checker.training_should_stop(model):
            print(f"NOT ENOUGH IMPROVEMENT FOUND, STOPPING TRAINING")
            validation_losses = validation_loss_checker.current_losses
            write_train_loss_to_file(epoch_list, train_losses, validation_losses)
            print(f"MODEL TO BE USED FROM EPOCH: {validation_loss_checker.epoch_of_saved_model}")

            model.load_state_dict(torch.load(f"{MODEL_NAME}/{MODEL_NAME}.pt"))
            validation_loss_checker.plot_pr_curve(model)

            return(train_losses, validation_losses, epoch_list)

        print(f"Epoch {epoch+1} complete. Final Avg Loss for this epoch: {epoch_average_loss}") # average loss across all batches
    
    save_model(model)

    validation_losses = validation_loss_checker.current_losses
    write_train_loss_to_file(epoch_list, train_losses, validation_losses)
    print(f"training completed")
    print(f"MODEL TO BE USED FROM EPOCH: {validation_loss_checker.epoch_of_saved_model}")

    return (train_losses, validation_losses, epoch_list)

def evaluate_model(model, test_data_loader):
    loss_function = BCEWithLogitsLoss()
    test_loss = 0
    total_num_of_predictions = 0
    number_of_correct_predictions = 0
    all_labels_across_batches = []
    all_predictions_across_batches = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_data_loader):
            images, labels = data
            outputs = model(images)
            loss = loss_function(outputs, labels)
            print(f"evaluation loss for batch {i+1}: {loss.item()}")

            predictions = (torch.sigmoid(outputs) > 0.5).int()

            all_labels_across_batches.extend(labels.numpy())
            all_predictions_across_batches.extend(predictions.numpy())


    report = classification_report(y_true=all_labels_across_batches, y_pred=all_predictions_across_batches, target_names=LABELS, output_dict=True)

    # save the report to a csv
    report_df = pd.DataFrame(report).transpose()
    
    report_path = Path(f'{MODEL_NAME}/evaluation/classification_report.csv')
    report_path.parent.mkdir(exist_ok=True, parents=True)
    
    report_df.to_csv(report_path) 
    print(report)


    confusion_matrix = multilabel_confusion_matrix(y_true=np.array(all_labels_across_batches), y_pred=np.array(all_predictions_across_batches))
    for i in range(len(LABELS)): # print the confusion matrix for the first class
        
        #labels_in_cm = np.unique(np.concatenate((np.array(all_labels_across_batches), np.array(all_predictions_across_batches))))
        #print(f"truth labels: {np.array(all_labels_across_batches).shape}")
        #print(f'predictions: {np.array(all_predictions_across_batches).shape}')
        #print(f'order of labels in the matrix: {labels_in_cm[0]}')
        matrix_plot = ConfusionMatrixDisplay(confusion_matrix[i])
        matrix_plot.plot()

        # save the plots
        matrix_filename = "confusion matrix " + LABELS[i]
        matrix_filepath = Path(f"{MODEL_NAME}/evaluation/confusion matrixes/"+matrix_filename)
        matrix_filepath.parent.mkdir(exist_ok=True, parents=True)
        matrix_plot.figure_.savefig(matrix_filepath)
        plt.close()

    return 


if __name__ == "__main__":
    # Just a test to check the module loads correctly initially
    augment_transforms = parse_arguments()[1]

    print(f"START TIME {datetime.now()}")

    # make the parent folder all of the logs, images, model will go into
    print(f"Evaluating Model {MODEL_NAME}")
    model_folder = Path(f"{MODEL_NAME}" )
    model_folder.mkdir(exist_ok=True, parents=True)

    if not os.path.exists("subset.csv"):
        print("subsetting data to create train and validation files")
        create_subset_train_validation()
    else:
        print("Train / Valid Subsets already created. Loader prep initializing..")

    if not os.path.exists("prepared_test.csv"):
        print("creating test dataset now")
        create_test_data_csv()

    dataloader_for_training, data_loader_for_validation = prepare_data(augment_transforms)
    data_loader_for_testing = prepare_test_data()

    class_weights = calculate_class_weights('train.csv')
    
    model = BaselineModel()
    if not os.path.exists(f'{MODEL_NAME}/{MODEL_NAME}.pt'):
        print("no previous models, training now")
        train_losses, validation_losses, epochs = train_model(model, dataloader_for_training, data_loader_for_validation)
        model.load_state_dict(torch.load(f"{MODEL_NAME}/{MODEL_NAME}.pt"))
        plot_loss(train_losses, validation_losses, epochs)
    else:
        print("previous models found!")
        model.load_state_dict(torch.load(f"{MODEL_NAME}/{MODEL_NAME}.pt"))

    post_train_model = model

    print('EVALUATION STARTING')
    evaluate_model(post_train_model, data_loader_for_testing)
    print(f"FINISH TIME {datetime.now()}")
