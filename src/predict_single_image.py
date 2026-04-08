from pathlib import Path
import torch
from constants.control_variables import MODEL_NAME, LABELS
from torchvision.transforms import Resize, Normalize, Compose, ToTensor
import cv2
from PIL import Image
from tabulate import *
import os
import json

def prepare_image_for_classification(image_path):
    """Takes a path to an image and classifies it using an existing model,
    specified by the --name parameter when running the file

    Args:
        image_path (string): path to image to classify

    Raises:
        TypeError: if the image provided does not exist, or is not a png, or jpg file, a type error is raised

    Returns:
        Tensor: image in the shape (1, 1, 224, 224) as the model expects a batch of images
    """
    mean = 0.5062857270240784
    standard_deviation = 0.2867498937006307

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


def make_classification(model, image, threshold):
    """Classifies a given image based on the chexpert labels
    Args: image_path [str]: The file path of the image to be classified

    Returns: table showing labels, the classification and confidence label
    """
    classifications, HEADERS = [], ["Label", "Prediction", "Confidence"]

    model.eval()
    print(f"Threshold used: {threshold}")
    with torch.no_grad():
        output = model(image)
        confidence_score = torch.sigmoid(
            output
        )  # turns the raw output into a range 0 - 1

        prediction = (confidence_score > torch.tensor(threshold)).int()

        for label, prediction, confidence_score in zip(
            LABELS, prediction.tolist()[0], confidence_score.tolist()[0]
        ):
            classifications.append([label, prediction, confidence_score])

    classification_table = tabulate(classifications, headers=HEADERS)

    return classification_table


def main():
    # Check if the model name they provided exists already or not
    path_to_model = input("Enter the path to the model you would like to run inference on: \n")

    if not os.path.exists(path_to_model):
        results_folder_exists = os.path.exists("results")
        error_msg = (
            f"Model not found at: {path_to_model}. "
            f"You could train the model using py main.py --name {os.path.basename(os.path.dirname(path_to_model))}"
            f"Then, return to this script, and provide the model path from the 'results' folder."
        )
        raise FileNotFoundError(error_msg)

    model = torch.load(path_to_model, weights_only=False)

    while True:
        try:
            image_path = input(
                "Please Enter the path of the image you would like to classify: \n"
            )
            image = prepare_image_for_classification(image_path)

            parent_of_model = Path(path_to_model).parent
            threshold_file = parent_of_model / "thresholds.json"
            if threshold_file.is_file():
                with open(threshold_file, 'r', encoding="utf-8") as threshold_file:
                    data = json.load(threshold_file)
                    threshold = data["thresholds"]

            else:
                threshold = [0.3]*13

            # if the model and image exists, produce the classification
            print(make_classification(model, image, threshold=threshold))

            break
        except TypeError as e:
            print(f"Error {e}. Try again.")

        except Exception as e:
            print(f"Something Unexpexted Went Wrong: {e}")


if __name__ == "__main__":
    main()
