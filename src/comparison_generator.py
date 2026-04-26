import matplotlib.pyplot as plt
import pandas as pd
from constants.control_variables import LABELS
import numpy as np
from pathlib import Path
from experiment_utils.arg_parser import parse_arguments
from RunPathHolder import RunPathHolder


def generate_comparisons(path_holder, save_path):
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

    # save_path = path_holder.comparison_path
    for model_dir in path_holder.experiment_root.iterdir():
        if model_dir.name == path_holder.comparison_path.name:
            continue
        model_dir_path_holder = RunPathHolder(
            experiment_name=path_holder.experiment_root, model_name=model_dir.name
        )
        model_name = model_dir_path_holder.model_name
        report = pd.read_csv(
            model_dir_path_holder.eval_data_path / "classification_report.csv",
            index_col=0,
        )

        macro_f1 = report.loc["macro avg","f1-score"]  # row macro avg and col f1-score
        macro_recall = report.loc["macro avg", "recall"]
        macro_ap = report.loc["macro avg", "Average Precision"]

        all_f1_scores[model_name] = macro_f1
        all_recall[model_name] = macro_recall
        all_ap[model_name] = macro_ap

        # set the index so i can use loc[] to loop through the labels
        # report = report.set_index("Label")
        for label in LABELS:
            label_recall = report.loc[label, "recall"]
            label_f1 = report.loc[label, "f1-score"]
            label_ap = report.loc[label, "Average Precision"]
            # the dict structure would be like this:
            # CLASS_LABEL: {Recall: {Model: 0.3, Model2: 0.3}}, {F1: {Model: 0, Model2: 0.9}}
            per_class_metrics[label]["recall"].update({model_name: label_recall})
            per_class_metrics[label]["f1"].update({model_name: label_f1})
            per_class_metrics[label]["ap"].update({model_name: label_ap})

    compare_f1(all_f1_scores, save_path)
    compare_recall(all_recall, save_path)
    compare_avg_precision(all_ap, save_path)

    compare_per_class_recall_f1_ap(per_class_metrics, save_path)

    return


def compare_f1(all_f1_scores, save_path_folder):
    """Creates a graph comparing macro average f1 score for each model

    Args:
        all_f1_scores (dict): key: model names | values: corresponding macro f1 score
    """

    x_model_names = all_f1_scores.keys()
    y_f1_scores = all_f1_scores.values()

    plt.clf()
    plt.figure(figsize=(15, 10))
    plt.barh(x_model_names, y_f1_scores)

    plt.xlabel("Experiment Name")
    plt.ylabel("Macro F1 Score")

    plt.savefig(save_path_folder / "F1_Scores.svg")
    plt.close()
    return


def compare_recall(all_recall, save_path_folder):
    """creates a graph of the recall values for all models from their classification reports

    Args:
        all_recall (dict): keys: model names, values: corresponding recall values
    """
    x_model_names = all_recall.keys()
    y_recall = all_recall.values()
    plt.clf()
    plt.figure(figsize=(15, 10))
    plt.barh(x_model_names, y_recall)

    plt.xlabel("Experiment Name")
    plt.ylabel("Macro Recall")

    plt.savefig(save_path_folder / "Recall.svg")
    plt.close()
    return


def compare_avg_precision(all_ap, save_path_folder):
    """Creates a graph comparing average precision score for all models

    Args:
        all_ap (dict): keys: model names | values: corresponding macro average_precision
    """
    x_model_names = all_ap.keys()
    y_ap = all_ap.values()

    plt.clf()
    plt.figure(figsize=(15, 10))
    plt.barh(x_model_names, y_ap)

    plt.xlabel("Experiment Name")
    plt.ylabel("Macro Average Precision")

    plt.savefig(save_path_folder / "Average_precision.svg")
    plt.close()
    return


def compare_per_class_recall_f1_ap(
    per_class_metrics: dict[str, dict[str, float]], save_path_folder
):
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
    print("-" * 20)
    print(per_class_metrics)

    # make the path if it exists already
    per_class_save_path = save_path_folder / "per-class-comparison"

    per_class_save_path.mkdir(exist_ok=True, parents=True)

    # grouped bar logic from: https://www.geeksforgeeks.org/python/create-a-grouped-bar-plot-in-matplotlib/
    for label in LABELS:
        x_model_name = per_class_metrics[label]["recall"].keys()
        y_recall = per_class_metrics[label]["recall"].values()
        y_f1 = per_class_metrics[label]["f1"].values()
        y_ap = per_class_metrics[label]["ap"].values()
        plt.clf()
        plt.figure(figsize=(25, 15))
        x_pos = np.arange(len(x_model_name))
        width = 0.2
        plt.bar(x_pos - 0.2, y_recall, width, color="blue")
        plt.bar(x_pos, y_f1, width, color="green")
        plt.bar(x_pos + 0.2, y_ap, width, color="yellow")
        plt.xticks(x_pos, x_model_name, fontsize=12, rotation=90)
        plt.xlabel("Model Name", fontsize=15)
        plt.ylabel("Score")
        plt.legend(["Recall", "F1 Score", "Average Precision"])
        plt.savefig(per_class_save_path / f"{label}-comparison.svg")
        plt.tight_layout()
        plt.close()
    return


def main():
    experiment_root, model_name, *_ = parse_arguments()
    path_holder = RunPathHolder(experiment_name=experiment_root, model_name=model_name)
    save_path = path_holder.comparison_path
    generate_comparisons(path_holder, save_path)
    print(f"Comparisons have been generated! See {save_path}")
    return


if __name__ == "__main__":
    main()
