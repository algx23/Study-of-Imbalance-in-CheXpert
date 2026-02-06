import torch
from constants import MODEL_NAME, LABELS
from path_creator import MODEL_ROOT
from torchvision.transforms import Resize, Normalize, Compose, ToTensor
import cv2
from model import BaselineModel
from PIL import Image
from tabulate import *
import os


def prepare_image_for_classification(image_path):
    mean = 0.5062857270240784
    standard_deviation = 0.2867498937006307

    TRANSFORMS = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean, standard_deviation)
                          ])

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise TypeError("No Valid image found at the path provided")

    image = Image.fromarray(image)
    image = TRANSFORMS(image)

    image = image.unsqueeze(0)

    return image

def make_classification(model, image): 
    ''' Classifies a given image based on the chexpert labels
    Args: image_path [str]: The file path of the image to be classified

    Returns: TODO
    '''
    classifications, HEADERS = [], ["Label", "Prediction", "Confidence"]

    model.eval()
    with torch.no_grad():
        output = model(image)
        confidence_score = torch.sigmoid(output) # turns the raw output into a range 0 - 1
        prediction = (confidence_score > 0.5).int()

        for label, prediction, confidence_score in zip(LABELS, prediction.tolist()[0], confidence_score.tolist()[0]):
            classifications.append([label, prediction, confidence_score])
    
    classification_table = tabulate(classifications, headers = HEADERS)
    
    return classification_table


def main():
    # Check if the model name they provided exists already or not
    path_to_model = MODEL_ROOT / f"{MODEL_NAME}.pt"

    if not os.path.exists(path_to_model):
        raise FileNotFoundError(f"Model {MODEL_NAME} not found at: {path_to_model}")

    model = torch.load(path_to_model, weights_only=False)

    while True:
        try:
            image_path = input("Please Enter the path of the image you would like to classify: \n")
            image = prepare_image_for_classification(image_path)

            # if the model and image exists, produce the classification
            print(make_classification(model, image))

            break
        except TypeError as e:
            print(f"Error {e}. Try again.")
            
        except Exception as e:
            print(f"Something Unexpexted Went Wrong: {e}")
        


if __name__ == "__main__":
    main()
