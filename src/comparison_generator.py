import os
import matplotlib.pyplot as plt
import pandas as pd
from constants.paths import COMPARISON_PATH
from constants.control_variables import LABELS
import numpy as np


def generate_comparisons(path_to_results):
    """Generates graphs comparing the F1 Score, Average Precision, and Recall for all trained models

    Args:
        path_to_results (string): the location of the results folder - all models/experiments are held in this directory
    """
    all_f1_scores = {}  # f1 score from report for each model -> macro
    all_recall = {}
    all_ap = {}
    per_class_metrics = {}

    for label in LABELS:
        per_class_metrics.update({label: {"recall": {}, "f1": {}, "ap": {}}})

    model_names = os.listdir(path_to_results)

    for model in model_names:
        report = pd.read_csv(
            f"{path_to_results}/{model}/evaluation/classification_report.csv",
            index_col=0,
        )
        print(report.index.tolist())

        macro_f1 = report.iloc[-3, -3]  # row macro avg and col f1-score
        macro_recall = report.iloc[-3, -4]
        macro_ap = report.iloc[-3, -1]
        
        

        all_f1_scores[model] = macro_f1
        all_recall[model] = macro_recall
        all_ap[model] = macro_ap


        # set the index so i can use loc[] to loop through the labels
        report = report.set_index("Label") 
        for label in LABELS:
            label_recall = report.loc[label, "recall"]
            label_f1 = report.loc[label, "f1-score"]
            label_ap = report.loc[label, "Average Precision"]
        # the dict structure would be like this:
        # CLASS_LABEL: {Recall: {Model: 0.3, Model2: 0.3}}, {F1: {Model: 0, Model2: 0.9}}
            per_class_metrics[label]["recall"].update({model: label_recall})
            per_class_metrics[label]["f1"].update({model: label_f1})
            per_class_metrics[label]["ap"].update({model: label_ap})

    compare_f1(all_f1_scores)
    compare_recall(all_recall)
    compare_avg_precision(all_ap)

    print(per_class_metrics)
    compare_per_class_recall_f1_ap(per_class_metrics)

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

def compare_per_class_recall_f1_ap(per_class_metrics: dict[str, dict[str, float]]):
    """
    Compare the F1, Recall and Average Precision (PR-AUC) of given classes
    across the different models, and group the metrics per model - i.e. each
    model will have 3 bars: one for Recall, one for F1 Score, and one for Average
    Precision.

    This will allow me to potentially make more in depth comparisons and have better
    insights as to which classes each model is better at, and the reasons why,
    compared to just comparing the macro averages. Additionally, graph creation is
    automated at the end of each evaluation run, so adding new models will mean an
    updated graph gets created.

    Args:
    - per_class_metrics: dict[str, dict[str, float]] : The dictionary containing
    the per class metrics for each class. The dictionary follows the structure:
    {class name : {Recall: {ModelName: val}}, {F1: {ModelName: val}, {AP: {ModelName: val}}}}
    """

    # TODO: Adapt this to make similar plots for other classes
    x_model_name = per_class_metrics["Fracture"]["recall"].keys()
    y_recall = per_class_metrics["Fracture"]["recall"].values()
    y_f1 = per_class_metrics["Fracture"]["f1"].values()
    y_ap = per_class_metrics["Fracture"]["ap"].values()
    print("-" * 20)
    print("\n")
    print(x_model_name)
    print(f"F1: {y_f1}")
    print(f"Recall: {y_recall}")
    print(f"ap: {y_ap}")

    # grouped bar logic from: https://www.geeksforgeeks.org/python/create-a-grouped-bar-plot-in-matplotlib/
    plt.clf()
    plt.figure(figsize=(30,15))
    x_pos = np.arange(len(x_model_name))
    width=0.2
    plt.bar(x_pos-0.2, y_recall, width, color="blue")
    plt.bar(x_pos, y_f1, width, color="green")
    plt.bar(x_pos+0.2, y_ap, width, color="yellow")
    plt.xticks(x_pos, x_model_name)
    plt.xlabel("Model Name")
    plt.ylabel("Score")
    plt.legend(["Recall", "F1 Score", "Average Precision"])
    plt.savefig(COMPARISON_PATH / "per-class-fracture.png")
    plt.close()
    return
