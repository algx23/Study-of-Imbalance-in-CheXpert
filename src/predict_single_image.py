from pathlib import Path
import torch
from constants.control_variables import LABELS
from torchvision.transforms import Resize, Normalize, Compose, ToTensor
import cv2
from PIL import Image
from tabulate import *
import os
import json

from constants.paths import TRAIN_SET_PATH
from data_preparation.loader_prep import prepare_data
from utils import calculate_mean_and_standard_deviation


def prepare_image_for_classification(image_path):
    """Takes a path to an image and classifies it using an existing model

    Args:
        image_path (string): path to image to classify

    Raises:
        TypeError: if the image provided does not exist, or is not a png, or jpg file, a type error is raised

    Returns:
        Tensor: image in the shape (1, 1, 224, 224) as the model expects a batch of images
    """
    norm_const_file_path = "norm_const.json"
    if os.path.exists(norm_const_file_path):
        with open(norm_const_file_path, "r") as norm_const_file:
            norm_const_file_data = json.load(norm_const_file)
            mean = norm_const_file_data["mean"]
            standard_deviation = norm_const_file_data["std"]
    else:
        # computed from a previous run - the norm const file contains the same values as it was computed on an earlier subset
        # but the difference is not that large, and since the mean/std is very similar whether it is calculated or not
        # this is the fallback - ideally i wouldve liked to have done all experiments with the calculated mean
        if Path(TRAIN_SET_PATH).is_file():
            train_dataloader = prepare_data(TRAIN_SET_PATH)
            mean, standard_deviation = calculate_mean_and_standard_deviation(
                train_dataloader, device
            )
        else:
            print(
                "The train dataset couldn't be found. Please make sure the CheXpert-V1-Small file is in the same location as this file. And that train.csv from the archive.zip is placed one directory above"
            )
            exit(1)

    TRANSFORMS = Compose(
        [Resize((224, 224)), ToTensor(), Normalize(mean, standard_deviation)]
    )

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise TypeError("No Valid image found at the path provided")

    image = Image.fromarray(image)
    image = TRANSFORMS(image)

    image = image.unsqueeze(0)

    return image


def make_classification(model, image, threshold, device):
    """Classifies a given image based on the chexpert labels
    Args:
        model: The model used to classify the image
        image: The image to be classified
        threshold: the threshold used to determine a positive or negative prediction
        device: the device to which the model, images and labels are moved (cpu or gpu)



    Returns: table showing labels, the classification and confidence label
    """
    classifications, HEADERS = [], [
        "Class Label",
        " Model Prediction",
        "Confidence that the image represents the positive class (%)",
    ]

    model.eval()
    print(
        f"Decision Thresholds used for each class - If the model is more confident than this %, they will predict positive {threshold}"
    )
    with torch.no_grad():
        output = model(image)
        confidence_score = torch.sigmoid(
            output
        )  # turns the raw output into a range 0 - 1

        prediction = (confidence_score > torch.tensor(threshold).to(device)).int()

        for label, prediction, confidence_score in zip(
            LABELS, prediction.tolist()[0], confidence_score.tolist()[0]
        ):
            classifications.append([label, prediction, confidence_score * 100])

    classification_table = tabulate(classifications, headers=HEADERS)

    return classification_table


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if the model name they provided exists already or not
    path_to_model = input(
        "Enter the path to the model you would like to run inference on: \n"
    )

    if not os.path.exists(path_to_model):
        error_msg = (
            f"Model not found at: {path_to_model}. Please make sure you have provided the full path to the model. E.g. From C:/ in windows."
            f"You could train the model using py main.py --name {os.path.basename(os.path.dirname(path_to_model))}"
            f"Then, return to this script, and provide the model path from the 'results' folder."
        )
        raise FileNotFoundError(error_msg)

    model = torch.load(path_to_model, weights_only=False)
    model = model.to(device)

    while True:
        try:
            image_path = input(
                "Please Enter the path of the image you would like to classify: \n"
            )
            image = prepare_image_for_classification(image_path)
            image = image.to(device)

            parent_of_model = Path(path_to_model).parent
            threshold_file_path = parent_of_model / "thresholds.json"
            if threshold_file_path.is_file():
                with open(threshold_file_path, "r", encoding="utf-8") as threshold_file:
                    data = json.load(threshold_file)
                    threshold = data["thresholds"]

            else:
                threshold = [0.3] * 13

            # if the model and image exists, produce the classification
            print(make_classification(model, image, threshold=threshold, device=device))

            break
        except TypeError as e:
            print(f"Error {e}. Try again.")

        except Exception as e:
            print(f"Something Unexpexted Went Wrong: {e}")


if __name__ == "__main__":
    main()
