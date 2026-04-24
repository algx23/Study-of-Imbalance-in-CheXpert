from pathlib import Path

import torch

from RunPathHolder import RunPathHolder
from experiment_utils import arg_parser

from metric_calculator import MetricCalculator


if __name__ == "__main__":
    experiment_root, model_name, *_ = arg_parser.parse_arguments()
    path_holder = RunPathHolder(experiment_name=experiment_root, model_name=model_name)

    truth_tensor_path = path_holder.tensor_save_path / "true_labels.pt"
    probability_tensor_path = path_holder.tensor_save_path / "probability_tensor.pt"
    prediction_tensor_path = path_holder.tensor_save_path / "prediction_tensor.pt"

    all_truth = torch.load(truth_tensor_path)
    all_probability = torch.load(probability_tensor_path)
    all_prediction = torch.load(prediction_tensor_path)

    metric_calculator = MetricCalculator(
        path_holder=path_holder,
        truth=all_truth,
        probabilities=all_probability,
        predictions=all_prediction,
    )

    metric_calculator.calculate_metrics()
    print("All Metrics Saved")
