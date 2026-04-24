from pathlib import Path


class RunPathHolder:
    """A class to hold the paths used throughout the project
    In previous iterations, I had the argument parsing through
    the constants file, and, this meant that it seemed clean
    but it meant that the imports got messy - some of my classes
    were importing from constants, but - eg i run my gradcam file
    seperately, asking for the path to the model, but if i ran py gradcam.py
    it would have needed arguments, because the argument parser always runs

    I also realized that the modelname, and vars for experiments, weren't control
    variables, so this class helped to move those to a place that made sense.
    """

    def __init__(self, experiment_name, model_name):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.experiment_root = Path(self.experiment_name)
        self.model_root = self.experiment_root / model_name
        self.train_data_path = self.model_root / "train"
        self.validation_data_path = self.model_root / "validation"
        self.eval_data_path = self.model_root / "evaluation"
        self.matrix_path = self.eval_data_path / "confusion matrices"
        self.comparison_path = self.experiment_root / "comparisons"
        self.tensor_save_path = self.eval_data_path / "tensor data"
        self.csv_paths = self.experiment_root.parent / "data"


def setup_folders(run_path_holder, csv_paths):

    csv_paths.mkdir(exist_ok=True, parents=True)
    run_path_holder.experiment_root.mkdir(exist_ok=True, parents=True)
    run_path_holder.model_root.mkdir(exist_ok=True, parents=True)
    run_path_holder.train_data_path.mkdir(exist_ok=True, parents=True)
    run_path_holder.validation_data_path.mkdir(exist_ok=True, parents=True)
    run_path_holder.eval_data_path.mkdir(exist_ok=True, parents=True)
    run_path_holder.matrix_path.mkdir(exist_ok=True, parents=True)
    run_path_holder.comparison_path.mkdir(exist_ok=True, parents=True)
    run_path_holder.tensor_save_path.mkdir(exist_ok=True, parents=True)
    return
