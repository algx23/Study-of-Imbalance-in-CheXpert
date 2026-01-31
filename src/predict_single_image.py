import torch
from constants import MODEL_NAME, LABELS
from torchvision.transforms import Resize, Normalize, Compose, ToTensor
import cv2
from model import BaselineModel
from PIL import Image
from tabulate import *


def make_classification(image_path: str): 
    ''' Classifies a given image based on the chexpert labels
    Args: image_path [str]: The file path of the image to be classified

    Returns: TODO
    '''
    mean =0.5062857270240784
    standard_deviation = 0.2867498937006307

    TRANSFORMS = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean, standard_deviation)
                          ])

    model = BaselineModel(use_dropout=False)
    model.load_state_dict(torch.load(f"{MODEL_NAME}/{MODEL_NAME}.pt"))

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = Image.fromarray(image)
    image = TRANSFORMS(image)

    image = image.unsqueeze(0)

    classifications, HEADERS = [], ["Label", "Prediction", "Confidence"]

    model.eval()
    with torch.no_grad():
        output = model(image)
        confidence_score = torch.sigmoid(output) # turns the raw output into a range 0 - 1
        prediction = (confidence_score > 0.5).int()

        for label, prediction, confidence_score in zip(LABELS, prediction.tolist()[0], confidence_score.tolist()[0]):
            classifications.append([label, prediction, confidence_score])

    
    print(tabulate(classifications, headers = HEADERS))
    
    return

image_path = input("Please Enter the path of the image you would like to classify: \n")

make_classification(image_path)
