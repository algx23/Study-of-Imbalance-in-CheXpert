import os
import matplotlib.pyplot as plt
import pandas as pd
from constants.paths import COMPARISON_PATH


def generate_comparisons(path_to_results):
    """Generates graphs comparing the F1 Score, Average Precision, and Recall for all trained models

    Args:
        path_to_results (string): the location of the results folder - all models/experiments are held in this directory
    """
    all_f1_scores = {}  # f1 score from report for each model -> macro
    all_recall = {}
    all_ap = {}

    model_names = os.listdir(path_to_results)

    for model in model_names:
        report = pd.read_csv(
            f"{path_to_results}/{model}/evaluation/classification_report.csv",
            index_col=0,
        )
        macro_f1 = report.iloc[-3, -3]  # row macro avg and col f1-score
        macro_recall = report.iloc[-3, -4]
        macro_ap = report.iloc[-3, -1]
        all_f1_scores[model] = macro_f1
        all_recall[model] = macro_recall
        all_ap[model] = macro_ap

    compare_f1(all_f1_scores)
    compare_recall(all_recall)
    compare_avg_precision(all_ap)

    return


def compare_f1(all_f1_scores):
    """Creates a graph comparing macro average f1 score for each model

    Args:
        all_f1_scores (dict): key: model names | values: corresponding macro f1 score
    """

    x_model_names = all_f1_scores.keys()
    y_f1_scores = all_f1_scores.values()

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.barh(x_model_names, y_f1_scores)
    # add the number on top of the bar
    # https://www.geeksforgeeks.org/python/adding-value-labels-on-a-matplotlib-bar-chart/
    for i in range(len(x_model_names)):
        plt.text(i, list(y_f1_scores)[i], round(list(y_f1_scores)[i], 5))
    plt.xlabel("Experiment Name")
    plt.ylabel("Macro F1 Score")

    plt.savefig(COMPARISON_PATH / "F1_Scores.png")
    plt.close()
    return


def compare_recall(all_recall):
    """creates a graph of the recall values for all models from their classification reports

    Args:
        all_recall (dict): keys: model names, values: corresponding recall values
    """
    x_model_names = all_recall.keys()
    y_recall = all_recall.values()
    plt.clf()
    plt.figure(figsize=(15,10))
    plt.barh(x_model_names, y_recall)
    # add the number on top of the bar
    # https://www.geeksforgeeks.org/python/adding-value-labels-on-a-matplotlib-bar-chart/
    for i in range(len(x_model_names)):
        plt.text(i, list(y_recall)[i], round(list(y_recall)[i], 5))
    plt.xlabel("Experiment Name")
    plt.ylabel("Macro Recall")

    plt.savefig(COMPARISON_PATH / "Recall.png")
    plt.close()
    return


def compare_avg_precision(all_ap):
    """Creates a graph comparing average precision score for all models

    Args:
        all_ap (dict): keys: model names | values: corresponding macro average_precision
    """
    x_model_names = all_ap.keys()
    y_ap = all_ap.values()

    plt.clf()
    plt.figure(figsize=(15,10))
    plt.barh(x_model_names, y_ap)
    # add the number on top of the bar
    # https://www.geeksforgeeks.org/python/adding-value-labels-on-a-matplotlib-bar-chart/
    for i in range(len(x_model_names)):
        plt.text(i, list(y_ap)[i], round(list(y_ap)[i], 5))

    plt.xlabel("Experiment Name")
    plt.ylabel("Macro Average Precision")

    plt.savefig(COMPARISON_PATH / "Average_precision.png")
    plt.close()
    return
