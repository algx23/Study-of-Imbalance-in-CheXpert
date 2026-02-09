import pandas as pd
import numpy as np
from constants.control_variables import LABELS, CHEXPERT_COMP_LABELS
from constants.paths import EVAL_DATA_PATH, MATRIX_PATH
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
)

import matplotlib.pyplot as plt


class MetricCalculator:
    """Generate Classification Report, Confusion Matrices, and Precision-Recall Curves"""

    def __init__(self, probabilities, predictions, truth):
        """initialize metric calculator

        Args:
            probabilities (np array): the sigmoid of the logits from the model
            predictions (np array): binary 0,1 class predictions for each class
            truth (np array): binary 0,1 truth labels for each class/image
        """
        self.probabilities = probabilities
        self.predictions = predictions
        self.truth = truth
        return

    def create_classification_report(self):
        """generate classification report"""

        report = classification_report(
            y_true=self.truth,
            y_pred=self.predictions,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        )

        # save the report to a csv
        # adapted from: https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format
        report_df = pd.DataFrame(report).transpose()

        # # https://stackoverflow.com/questions/19851005/rename-pandas-dataframe-index
        report_df.index.name = "Label"  # to change 'the unamed: 0' column to label

        report_path = EVAL_DATA_PATH / "classification_report.csv"
        report_df.to_csv(report_path)
        print(report)
        return

    def create_per_class_confusion_matrices(self):
        """Generate a confusion matrix for each class
        Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
        """
        confusion_matrices = multilabel_confusion_matrix(
            y_true=np.array(self.truth), y_pred=np.array(self.predictions)
        )
        for i in range(len(LABELS)):

            matrix_plot = ConfusionMatrixDisplay(confusion_matrices[i])
            matrix_plot.plot()

            # save the plots
            matrix_filepath = MATRIX_PATH / f"{LABELS[i]}_confusion_matrix.png"
            matrix_plot.figure_.savefig(matrix_filepath)

        return

    def plot_pr_curve(self):
        """Plot PR curve and save to a file, and calculate average precision and add to the classification report
        Reference:
            - PR Curves: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
            - Changing the figure to control where its plotted:
                https://stackoverflow.com/questions/7986567/matplotlib-how-to-set-the-current-figure
        """
        all_pr_curve_fig = plt.figure(figsize=(10, 6))
        comp_pr_curve_fig = plt.figure(figsize=(10, 6))
        avg_precision_scores = []

        for i in range(len(LABELS)):
            if LABELS[i] in CHEXPERT_COMP_LABELS:
                plt.figure(comp_pr_curve_fig.number)
                precision, recall, thresholds = precision_recall_curve(
                    np.array(self.truth)[:, i], np.array(self.probabilities)[:, i]
                )
                plt.plot(recall, precision, label=f"{LABELS[i]}")

            plt.figure(all_pr_curve_fig.number)
            precision, recall, thresholds = precision_recall_curve(
                np.array(self.truth)[:, i], np.array(self.probabilities)[:, i]
            )
            plt.plot(recall, precision, label=f"{LABELS[i]}")

            # compute the average precision for each label
            # and add it as a column to the classification report
            avg_precision = average_precision_score(
                y_true=np.array(self.truth)[:, i],
                y_score=np.array(self.probabilities)[:, i],
            )

            avg_precision_scores.append(avg_precision)

        # compute the average precision scores also to match the rest of the classsification report
        averages_for_precision = ["micro", "macro", "weighted", "samples"]
        for average in averages_for_precision:
            ap = average_precision_score(
                y_true=np.array(self.truth),
                y_score=np.array(self.probabilities),
                average=average,
            )
            avg_precision_scores.append(ap)

        # add the average precisions to the classification report
        report = pd.read_csv(EVAL_DATA_PATH / "classification_report.csv")
        report["Average Precision"] = pd.Series(avg_precision_scores)
        report.to_csv(EVAL_DATA_PATH / "classification_report.csv")

        # format+save the pr curve figure of all classes
        plt.figure(all_pr_curve_fig.number)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt_save_path = f"{EVAL_DATA_PATH}/pr_curve.png"
        plt.savefig(plt_save_path, bbox_inches="tight")

        # format+save the competition pr curve figure
        plt.figure(comp_pr_curve_fig.number)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(bbox_to_anchor=(1.05, 1))

        plt_save_path = f"{EVAL_DATA_PATH}/comp_pr_curve.png"
        plt.savefig(plt_save_path, bbox_inches="tight")

        return

    def calculate_metrics(self):
        """entry point to calculate all metrics"""
        self.create_classification_report()
        self.create_per_class_confusion_matrices()
        self.plot_pr_curve()
        return
