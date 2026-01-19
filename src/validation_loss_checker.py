from torch.nn import BCEWithLogitsLoss
import torch
import math
from utils import save_model
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from constants import MODEL_NAME
from pathlib import Path

class ValidationLossChecker():
    def __init__(self, min_improvement, epochs_to_wait, validation_loader):
        self.current_losses = []
        self.epoch_of_saved_model = 0

        self.min_improvement = min_improvement
        self.epochs_to_wait = epochs_to_wait
        self.num_epochs_no_gain = 0

        self.validation_loader = validation_loader

        self.stop_early = False
        self.best_loss = math.inf
        self.loss_function = BCEWithLogitsLoss()

    def compute_validation_loss(self, model):
        all_predictions, all_labels = [], []
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, data in enumerate(self.validation_loader):
                images, labels = data
                outputs = model(images)
                loss = self.loss_function(outputs, labels)
                total_loss += loss.item()                

            loss_for_epoch = total_loss / len(self.validation_loader)

        return loss_for_epoch

    def plot_pr_curve(self, model):
        all_predictions, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.validation_loader):
                images, labels = data
                outputs = model(images)
                predictions = torch.sigmoid(outputs)
                all_predictions.extend(predictions.numpy())
                all_labels.extend(labels.numpy())

        for i in range(13):
            precision, recall, thresholds = precision_recall_curve(np.array(all_labels)[:, i], np.array(all_predictions)[:, i])
            plt.plot(recall, precision, label=f'Class {i}')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()

        plt_save_path = Path(f"{MODEL_NAME}/{MODEL_NAME}_pr_curve.png")
        plt_save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(plt_save_path)
        plt.clf()
        return

    def check_for_no_improvement(self, model):
        current_loss = self.compute_validation_loss(model)
        self.current_losses.append(current_loss)

        # a better loss is lower than the current, by at least the min
        # improvement amount
        if current_loss < self.best_loss - self.min_improvement:
            #print("WE FOUND A NEW LOSS")
            self.best_loss = current_loss
            # if there is a improvement, reset the counter
            self.num_epochs_no_gain = 0
            save_model(model) # save the model with the best loss
            self.epoch_of_saved_model += 1
        else: # not enough gain to constitute a new best loss
            self.num_epochs_no_gain += 1

        print(f"best loss {self.best_loss}, current loss: {current_loss}")
        print(f"Epcohs without Improvement {self.num_epochs_no_gain} / {self.epochs_to_wait}")
        return

    def training_should_stop(self, model):
        self.check_for_no_improvement(model)
        if self.num_epochs_no_gain >= self.epochs_to_wait:
            self.stop_early = True

        return self.stop_early
