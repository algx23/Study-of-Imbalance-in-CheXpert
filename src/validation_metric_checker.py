from torch.nn import BCEWithLogitsLoss
import torch
from utils import save_model
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_curve
import numpy as np


class ValidationMetricCalculator:
    def __init__(
        self,
        best_model_save_path,
        min_improvement,
        epochs_to_wait,
        validation_loader,
        loss_fn,
    ):
        """Initialze the loss checker to check loss on validation set at the end of every epoch

        Args:
            best_model_save_path (str): path to save the model with the best validation Average Precision to
            min_improvement (float): minimum reduction in validation loss to count as an improvement
            epochs_to_wait (int): number of epochs where no improvement by the min_improvement is acceptable before training stops
            validation_loader (DataLoader): Dataloader of the validation set
            class_weights (Tensor): class weights for use in the loss function
        """
        self.current_metrics = []
        self.epoch_of_saved_model = 0

        self.best_model_save_path = best_model_save_path
        self.min_improvement = min_improvement
        self.epochs_to_wait = epochs_to_wait
        self.num_epochs_no_gain = 0

        self.validation_loader = validation_loader

        self.stop_early = False
        self.best_metric = 0
        self.loss_function = loss_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_validation_metric(self, model):
        """Compute the validation loss on the validation set with the given model

        Args:
            model (BaselineModel): The model to be validated

        Returns:
            float: the validation average precision for the epoch
        """
        all_probabilities = []
        all_truth = []
        all_predictions = []

        model.to(self.device)
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.validation_loader):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)

                all_truth.extend(labels.cpu().numpy())

                probability = torch.sigmoid(outputs)
                all_probabilities.extend(probability.cpu().numpy())

                prediction = (probability > 0.3).int()
                all_predictions.extend(prediction.cpu().numpy())

        validation_pr_auc = average_precision_score(
            y_true=all_truth, y_score=all_probabilities
        )

        return validation_pr_auc

    def check_for_no_improvement(self, model):
        """Check the current loss against the best loss and update
        the best loss if there is a big enough improvement

        Args:
            model (BaselineModel): The model on which to check the current loss against the best validation loss
        """
        current_metric = self.compute_validation_metric(model)
        self.current_metrics.append(current_metric)

        # a better loss is lower than the current, by at least the min
        # improvement amount
        if current_metric > self.best_metric + self.min_improvement:
            self.best_metric = current_metric
            # if there is a improvement, reset the counter
            self.num_epochs_no_gain = 0
            save_model(
                model, save_path=self.best_model_save_path
            )  # save the model with the best loss
            self.epoch_of_saved_model += 1
        else:  # not enough gain to constitute a new best loss
            self.num_epochs_no_gain += 1

        print(f"best prauc {self.best_metric}, current prauc: {current_metric}")
        print(
            f"Epcohs without Improvement {self.num_epochs_no_gain} / {self.epochs_to_wait}"
        )
        return

    def training_should_stop(self, model):
        """Decide whether training should stop early

        Args:
            model (BaselineModel): The model being trained

        Returns:
            bool: True if no improvements have been made for at least epochs_to_wait
        """
        self.check_for_no_improvement(model)
        if self.num_epochs_no_gain >= self.epochs_to_wait:
            self.stop_early = True

        return self.stop_early

    def calculate_optimal_threshold(self, model):
        """Calculates the Optimal Threshold of a model, by optimizing the F1 Score on the precision-recall curve

        Args:
            model (BaselineModel): the model to calculate optimal thresholds for

        Returns:
            list[float]: the list of optimal thresholds for each class
        """
        all_probabilities = []
        all_truth = []
        all_predictions = []

        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.validation_loader):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)

                all_truth.extend(labels.cpu().numpy())

                probability = torch.sigmoid(outputs)
                all_probabilities.extend(probability.cpu().numpy())

                prediction = (probability > 0.3).int()
                all_predictions.extend(prediction.cpu().numpy())

        best_thresholds = []
        # youden's jscore adapted from: https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
        # f1 score maximization to find the best threshold
        # https://www.sciencedirect.com/science/article/pii/S2214579615000611
        for i in range(13):
            precision, recall, thresholds = precision_recall_curve(
                y_true=np.array(all_truth)[:, i],
                y_score=np.array(all_probabilities)[:, i],
            )
            f1_scores = (2 * recall * precision) / (
                recall + precision + 1e-7
            )  # 1e-7 in case its 0
            index_of_max_f1_score = np.argmax(f1_scores)
            best_threshold_for_class = thresholds[index_of_max_f1_score]
            best_thresholds.append(float(best_threshold_for_class))

        return best_thresholds
