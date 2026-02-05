from path_creator import EVAL_DATA_PATH, MATRIX_PATH
from constants import LABELS
import pandas as pd
import torch
from sklearn.metrics import (classification_report, 
                             multilabel_confusion_matrix,
                             ConfusionMatrixDisplay)
from pathlib import Path
import numpy as np
from metric_calculator import MetricCalculator
                             

class EvaluationLoop():

    def __init__(self, test_loader, model, loss_fn):

        self.test_loader = test_loader
        self.model = model
        self.loss_fn = loss_fn

    def evaluate_model(self):
        # save prediction/ground truth tensors to file for future logging
        tensor_save_path = Path(f"{EVAL_DATA_PATH}/tensor_data")
        tensor_save_path.mkdir(exist_ok=True, parents=True)

        test_loss = 0
        total_num_of_predictions = 0
        number_of_correct_predictions = 0
        all_labels_across_batches = []
        all_predictions_across_batches = []
        all_outputs = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                images, labels = data
                outputs = self.model(images)
                all_outputs.extend(outputs.data.numpy())
                loss = self.loss_fn(outputs, labels)
                print(f"evaluation loss for batch {i+1}: {loss.item()}")

                predictions = (torch.sigmoid(outputs) > 0.5).int()

                all_labels_across_batches.extend(labels.numpy())
                all_predictions_across_batches.extend(predictions.numpy())



        torch.save(all_labels_across_batches, f"{tensor_save_path}/truth_tensor.pt")
        torch.save(all_predictions_across_batches, f"{tensor_save_path}/prediction_tensor.pt")

        logit_df = pd.DataFrame(all_outputs, columns=LABELS)
        logit_df.to_csv(EVAL_DATA_PATH / "eval_logits.csv")

        metric_calculator = MetricCalculator(predictions=all_predictions_across_batches, truth=all_labels_across_batches)
        metric_calculator.calculate_metrics()
        return
