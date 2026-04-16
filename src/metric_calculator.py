from pathlib import Path
import traceback
import warnings

import pandas as pd
import numpy as np
from RunPathHolder import RunPathHolder
from constants.control_variables import (
    LABELS,
    CHEXPERT_COMP_LABELS,
    RARE_CHEXPERT_LABELS,
)
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
)

import matplotlib.pyplot as plt
import seaborn as sns

from utils import plot_loss_ap


class MetricCalculator:
    """Generate Classification Report, Confusion Matrices, and Precision-Recall Curves"""

    def __init__(
        self,
        path_holder: RunPathHolder,
        probabilities,
        predictions,
        truth,
    ):
        """initialize metric calculator

        Args:
            probabilities (np array): the sigmoid of the logits from the model
            predictions (np array): binary 0,1 class predictions for each class
            truth (np array): binary 0,1 truth labels for each class/image
        """
        self.path_holder = path_holder
        self.probabilities = np.array(probabilities).astype(float)
        self.predictions = np.array(predictions).astype(int)

        self.truth = np.array(truth).astype(int)
        return

    def plot_train_loss_and_val_prauc(self):
        train_graph_path = (
            self.path_holder.train_data_path / "avg_train_loss_val_prauc_per_epoch.csv"
        )
        loss_prauc_df = pd.read_csv(train_graph_path)
        epoch, train_loss, val_prauc = (
            loss_prauc_df["Epoch"],
            loss_prauc_df["Train Loss"],
            loss_prauc_df["Validation Average Precision"],
        )
        train_plot_save_path = (
            self.path_holder.train_data_path
            / f"Train Loss vs Validation PR-AUC for {self.path_holder.model_name}"
        )
        plot_loss_ap(
            training_losses=train_loss,
            validation_ap=val_prauc,
            epochs=epoch,
            save_path=train_plot_save_path,
        )

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

        avg_precision_scores = []

        for i in range(len(LABELS)):
            # compute the average precision for each label
            # and add it as a column to the classification report
            print(self.truth[:, i].shape)
            avg_precision = average_precision_score(
                y_true=self.truth[:, i].astype(int),
                y_score=self.probabilities[:, i].astype(float),
            )

            avg_precision_scores.append(avg_precision)

            # compute the average precision scores also to match the rest of the classsification report
        averages_for_precision = ["micro", "macro", "weighted", "samples"]
        for average in averages_for_precision:
            ap = average_precision_score(
                y_true=self.truth,
                y_score=self.probabilities,
                average=average,
            )
            avg_precision_scores.append(ap)

        # add the average precisions to the classification report
        report["Average Precision"] = pd.Series(avg_precision_scores)

        report_path = self.path_holder.eval_data_path / "classification_report.csv"
        report_df.to_csv(report_path)
        print(report)
        return

    def create_per_class_confusion_matrices(self):
        """Generate a confusion matrix for each class
        Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
        """
        combined_matrix_path = self.path_holder.matrix_path / "Complete matrices.svg"
        confusion_matrices = multilabel_confusion_matrix(
            y_true=np.array(self.truth), y_pred=np.array(self.predictions)
        )

        # combined subplots, and individual matrix
        combined_fig, combined_axis = plt.subplots(nrows=4, ncols=4, figsize=(20, 16))
        combined_fig.suptitle(
            f"{self.path_holder.model_name.replace("_", " ")} Confusion Matrices",
            fontsize=14,
        )
        flattened_combined_axis = combined_axis.flat
        for i in range(len(LABELS)):

            matrix_plot = ConfusionMatrixDisplay(confusion_matrices[i])
            matrix_plot.plot(
                ax=flattened_combined_axis[i], cmap="Blues", colorbar=False
            )
            flattened_combined_axis[i].set_title(f"{LABELS[i]}", fontsize=10)

        for ax in flattened_combined_axis[len(LABELS) :]:
            ax.remove()

        # save the plots
        combined_fig.tight_layout()

        combined_fig.savefig(combined_matrix_path)
        plt.close(combined_fig)

        for i in range(len(LABELS)):
            matrix_save_path = (
                self.path_holder.matrix_path / f"{LABELS[i]}-Confusion-Matrix.svg"
            )
            matrix_plot = ConfusionMatrixDisplay(confusion_matrices[i])
            matrix_plot.plot(cmap="Blues", colorbar=False)
            matrix_plot.ax_.set_title(f"{LABELS[i]}-Confusion-Matrix")
            matrix_plot.figure_.savefig(
                matrix_save_path,
            )
        plt.close()

        return

    def plot_pr_curves(self):
        """Plot PR curve and save to a file, and calculate average precision and add to the classification report
        Reference:
            - PR Curves: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
            - Changing the figure to control where its plotted:
                https://stackoverflow.com/questions/7986567/matplotlib-how-to-set-the-current-figure
        """
        all_pr_curve_fig = plt.figure(figsize=(10, 6))
        comp_pr_curve_fig = plt.figure(figsize=(10, 6))
        low_count_pr_curve_fig = plt.figure(figsize=(10, 6))
        colour = plt.get_cmap("tab20")

        for i in range(len(LABELS)):
            print(
                f"Label: {LABELS[i]} | Unique Truth: {np.unique(self.truth[:, i])} | Sum: {self.truth[:, i].sum()}"
            )
            if LABELS[i] in CHEXPERT_COMP_LABELS:
                plt.figure(comp_pr_curve_fig.number)
                precision, recall, thresholds = precision_recall_curve(
                    self.truth[:, i], self.probabilities[:, i]
                )
                plt.plot(recall, precision, label=f"{LABELS[i]}", color=colour(i))

            if LABELS[i] in RARE_CHEXPERT_LABELS:
                plt.figure(low_count_pr_curve_fig.number)
                precision, recall, thresholds = precision_recall_curve(
                    self.truth[:, i], self.probabilities[:, i]
                )

                plt.plot(recall, precision, label=f"{LABELS[i]}", color=colour(i))

            plt.figure(all_pr_curve_fig.number)

            precision, recall, thresholds = precision_recall_curve(
                self.truth[:, i], self.probabilities[:, i]
            )
            plt.plot(recall, precision, label=f"{LABELS[i]}", color=colour(i))

        # format+save the pr curve figure of all classes
        plt.figure(all_pr_curve_fig.number)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt_save_path = self.path_holder.eval_data_path / "pr_curve.svg"
        plt.savefig(plt_save_path, bbox_inches="tight")

        # format+save the competition pr curve figure
        plt.figure(comp_pr_curve_fig.number)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(bbox_to_anchor=(1.05, 1))

        plt_save_path = self.path_holder.eval_data_path / "comp_pr_curve.svg"
        plt.savefig(plt_save_path, bbox_inches="tight")

        # format+save the low class pr curve figure
        plt.figure(low_count_pr_curve_fig.number)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(bbox_to_anchor=(1.05, 1))

        plt_save_path = self.path_holder.eval_data_path / "low-support-classess.svg"
        plt.savefig(plt_save_path, bbox_inches="tight")
        plt.close()

        return

    def compute_co_occurance_matrix_between_prediction_and_truth(self):
        co_occurance = np.dot(np.array(self.truth).T, np.array(self.predictions))
        # normalize: https://stackoverflow.com/questions/8904694/how-to-normalize-a-2-dimensional-numpy-array-in-python-less-verbose
        cooc_row_sum = co_occurance.sum(axis=1)
        normalized_cooc_row_sum = co_occurance / cooc_row_sum[:, np.newaxis]

        sns.heatmap(
            normalized_cooc_row_sum,
            cmap="coolwarm",
            annot=True,
            xticklabels=LABELS,
            yticklabels=LABELS,
            cbar=False,
            fmt=".2f",
        )

        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title(
            f"Prediction Co-Occurance Matrix for {self.path_holder.model_name.replace("_", " ")}"
        )
        cooc_save_path = (
            self.path_holder.eval_data_path
            / f"{self.path_holder.model_name.replace("_", " ")} Co-Occurance Matrix.svg"
        )
        plt.savefig(cooc_save_path)
        plt.close()
        return

    def calculate_metrics(self):
        """entry point to calculate all metrics"""
        self.plot_train_loss_and_val_prauc()
        self.create_classification_report()
        self.create_per_class_confusion_matrices()
        self.plot_pr_curves()
        self.compute_co_occurance_matrix_between_prediction_and_truth()
        return
