from constants.control_variables import LABELS
import pandas as pd
import torch
from pathlib import Path
import numpy as np


class EvaluationLoop:
    """The Evaluation loop to test the performance of all models, and generate metrics for comparison"""

    def __init__(self, path_holder, test_loader, model, loss_fn):
        """Initialize the evaluation loop

        Args:
            test_loader (DataLoader): the dataloader on which to test
            model (BaselineModel): the model instance to test
            loss_fn (BCEWithLogitsLoss): the loss function to use
        """

        self.path_holder = path_holder
        self.test_loader = test_loader
        self.model = model
        self.loss_fn = loss_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_model(self, thresholds):
        """Evalues the model, saves logits and calculates metrics"""
        # save prediction/ground truth tensors to file for future logging

        all_labels_across_batches = []
        all_predictions_across_batches = []
        all_outputs = []
        all_probabilities = []

        self.model.eval()

        print(f"thresholds used : {thresholds}")
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                all_outputs.extend(
                    outputs.cpu().numpy()
                )  # needs to be moved to cpu on appending here because i save it later
                loss = self.loss_fn(outputs, labels)
                print(f"evaluation loss for batch {i+1}: {loss.item()}")
                probability = torch.sigmoid(outputs)
                all_probabilities.extend(probability.cpu().numpy())

                predictions = (
                    probability > torch.tensor(thresholds).to(self.device)
                ).int()

                all_labels_across_batches.extend(labels.cpu().numpy())
                all_predictions_across_batches.extend(predictions.cpu().numpy())

        predictions_save_path = (
            self.path_holder.tensor_save_path / "prediction_tensor.pt"
        )

        truth_label_save_path = self.path_holder.tensor_save_path / "true_labels.pt"

        probability_save_path = (
            self.path_holder.tensor_save_path / "probability_tensor.pt"
        )

        torch.save(
            torch.tensor(all_labels_across_batches),
            truth_label_save_path,
        )
        torch.save(
            torch.tensor(all_predictions_across_batches),
            predictions_save_path,
        )

        torch.save(torch.tensor(all_probabilities), probability_save_path)

        # saving the logits just in case
        logit_df = pd.DataFrame(all_outputs, columns=LABELS)
        logit_df_path = self.path_holder.eval_data_path / "eval_logits.csv"
        logit_df.to_csv(logit_df_path)

        return
