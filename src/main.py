import os
import pandas as pd
from datetime import datetime
from experiment_utils.arg_parser import parse_arguments

from sample_dataset_to_5000 import DataSubsetter

import torch
from chexpert_dataset import ChexpertDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import Compose,Normalize, Resize, ToTensor # to resize all images
from utils import (calculate_mean_and_standard_deviation,
                   calculate_class_weights,
                   save_model,
                   write_train_loss_to_file)


from model import BaselineModel
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from constants import (
    NUM_EPOCHS,
    LABELS,
    MODEL_NAME,
    IMAGES_PATH,
    ORIGINAL_DATASET_PATH,
    SUBSET_PATH,
    TRAIN_SET_PATH,
    VALIDATION_SET_PATH,
    TEST_SET_PATH
)
from validation_loss_checker import ValidationLossChecker

import matplotlib.pyplot as plt
from utils import plot_loss

from pathlib import Path
from path_creator import (
    setup_folders,
    MODEL_ROOT,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    EVAL_DATA_PATH,
    MATRIX_PATH
)

from sklearn.metrics import (multilabel_confusion_matrix,
                             classification_report,
                             roc_auc_score,
                             ConfusionMatrixDisplay)
import numpy as np
from PIL import Image

# augmentation imports
from torchvision.transforms import (RandomRotation,
                                    RandomHorizontalFlip)


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
    test_dataset = ChexpertDataset("prepared_test.csv", "D:/dataset fyp/", transform=resize_transform)

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

    after_normalization_test_dataset = ChexpertDataset("prepared_test.csv", "D:/dataset fyp/", transform=transforms)
    after_normalization_test_loader = DataLoader(after_normalization_test_dataset, batch_size=25, shuffle=True)

    return after_normalization_test_loader


def train_model(model, train_dataloader, validation_dataloader, class_weights=None):

    validation_loss_checker = ValidationLossChecker(
        min_improvement=0.0001,
        epochs_to_wait=10,
        validation_loader = validation_dataloader,
        class_weights=class_weights
    )
    loss_function = BCEWithLogitsLoss(pos_weight=class_weights)
    print(f"Loss Weights: {loss_function.pos_weight}")
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

            model.load_state_dict(torch.load({MODEL_ROOT} / f"{MODEL_NAME}.pt"))
            validation_loss_checker.plot_pr_curve(model)

            return(train_losses, validation_losses, epoch_list)

        print(f"Epoch {epoch+1} complete. Final Avg Loss for this epoch: {epoch_average_loss}") # average loss across all batches
    
    save_model(model)

    validation_losses = validation_loss_checker.current_losses
    write_train_loss_to_file(epoch_list, train_losses, validation_losses)

    validation_loss_checker.plot_pr_curve(model)

    print(f"training completed")
    # epoch_list[-1] shows the last epoch that was completed and epoch_of_saved_model actually shows the num of improvements
    # so epoch_list[-1] - epoch_of_saved_model is the last epoch a model was saved
    # TODO: rewrite/rename so this makes more sense
    print(f"MODEL TO BE USED FROM EPOCH: {epoch_list[-1] - validation_loss_checker.epoch_of_saved_model}")

    return (train_losses, validation_losses, epoch_list)

def evaluate_model(model, test_data_loader):
    # save prediction/ground truth tensors to file for future logging
    tensor_save_path = Path(f"{MODEL_NAME}/evaluation/tensor_data")
    tensor_save_path.mkdir(exist_ok=True,parents=True)

    loss_function = BCEWithLogitsLoss()
    test_loss = 0
    total_num_of_predictions = 0
    number_of_correct_predictions = 0
    all_labels_across_batches = []
    all_predictions_across_batches = []
    all_outputs = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_data_loader):
            images, labels = data
            outputs = model(images)
            all_outputs.extend(outputs.data.numpy())
            loss = loss_function(outputs, labels)
            print(f"evaluation loss for batch {i+1}: {loss.item()}")

            predictions = (torch.sigmoid(outputs) > 0.5).int()

            all_labels_across_batches.extend(labels.numpy())
            all_predictions_across_batches.extend(predictions.numpy())



    torch.save(all_labels_across_batches, f"{tensor_save_path}/truth_tensor.pt")
    torch.save(all_predictions_across_batches, f"{tensor_save_path}/prediction_tensor.pt")

    logit_df = pd.DataFrame(all_outputs, columns=LABELS)
    logit_df.to_csv(EVAL_DATA_PATH / "eval_logits.csv")

    report = classification_report(y_true=all_labels_across_batches, y_pred=all_predictions_across_batches, target_names=LABELS, output_dict=True)

    # save the report to a csv
    report_df = pd.DataFrame(report).transpose()
    
    report_path = EVAL_DATA_PATH / "classification_report.csv"
    report_df.to_csv(report_path) 
    print(report)

    confusion_matrix = multilabel_confusion_matrix(y_true=np.array(all_labels_across_batches), y_pred=np.array(all_predictions_across_batches))
    for i in range(len(LABELS)): # print the confusion matrix for the first class
        
        matrix_plot = ConfusionMatrixDisplay(confusion_matrix[i])
        matrix_plot.plot()

        # save the plots
        matrix_filepath = MATRIX_PATH / f"{LABELS[i]}_confusion_matrix"
        matrix_plot.figure_.savefig(matrix_filepath)

    return 


if __name__ == "__main__":
    setup_folders()

    _, augment_transforms, use_weights, use_clahe, use_dropout, use_batch_norm = parse_arguments() 

    # make the parent folder all of the logs, images, model will go into
    print(f"Evaluating Model {MODEL_NAME}")
    print(f"CLASS WEIGHTS USED {use_weights}")
    print(f"BATCH NORM USED:  {use_batch_norm}")
    print(f"START TIME {datetime.now()}")

    subsetter = DataSubsetter(ORIGINAL_DATASET_PATH, TRAIN_SET_PATH, VALIDATION_SET_PATH, TEST_SET_PATH, LABELS)

    if not (os.path.exists(TRAIN_SET_PATH)
            and os.path.exists(VALIDATION_SET_PATH)):
        print("subsetting data to create train and validation files")
        subsetter.create_subset_train_validation()
    else:
        print("Train / Valid Subsets already created. Loader prep initializing..")

    if not os.path.exists(TEST_SET_PATH):
        print("creating test dataset now")
        subsetter.create_test_data_csv()

    dataloader_for_training, data_loader_for_validation = prepare_data(augment_transforms, use_clahe, IMAGES_PATH, TRAIN_SET_PATH, VALIDATION_SET_PATH)
    data_loader_for_testing = prepare_test_data(TEST_SET_PATH)

    class_weights = calculate_class_weights(TRAIN_SET_PATH) if use_weights else None
    
    print(f"USE DROPOUT: {use_dropout}")
    model = BaselineModel(use_dropout=use_dropout, use_batch_norm=use_batch_norm)

    if not os.path.exists(f'{MODEL_ROOT}/{MODEL_NAME}.pt'):
        print("no previous models, training now")
        train_losses, validation_losses, epochs = train_model(model, dataloader_for_training, data_loader_for_validation, class_weights)
        model.load_state_dict(torch.load(MODEL_ROOT / f"{MODEL_NAME}.pt"))
        plot_loss(train_losses, validation_losses, epochs)
    else:
        print("previous models found!")
        model.load_state_dict(torch.load(MODEL_ROOT / f"{MODEL_NAME}.pt"))

    post_train_model = model

    print('EVALUATION STARTING')
    evaluate_model(post_train_model, data_loader_for_testing)
    print(f"FINISH TIME {datetime.now()}")

