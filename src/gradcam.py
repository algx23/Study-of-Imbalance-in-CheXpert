import os
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
import torch
from PIL import Image
from constants.control_variables import LABELS
import json
from pathlib import Path

import cv2

import numpy as np

import matplotlib.pyplot as plt
from utils import calculate_mean_and_standard_deviation


class GradCam:
    """
    A GradCam Implementation to generate a heatmap of activations
    for a given model.

    References:
        - Original GradCam paper: https://arxiv.org/pdf/1610.02391
        - Adapted from TowardsDataScience: https://towardsdatascience.com/grad-cam-from-scratch-with-pytorch-hooks/
    """

    def __init__(self, model, model_name, image_path, mean, std):
        self.model = model

        self.mean = mean
        self.standard_deviation = std
        self.transforms = Compose(
            [
                Resize((224, 224)),
                ToTensor(),
                Normalize(self.mean, self.standard_deviation),
            ]
        )
        self.activations = []
        self.gradients = []
        self.model_name = model_name
        self.image_path = image_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return

    def compute_gradcam_heatmap(self):
        self.model.to(self.device)
        self.model.eval()
        self.model.zero_grad()
        image = Image.open(self.image_path)
        image = self.transforms(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        fwd_hook = self.model.conv_2.register_forward_hook(self.save_activations)
        bwd_hook = self.model.conv_2.register_full_backward_hook(self.save_gradient)

        prediction = self.model(image)
        target = torch.argmax(prediction[0])
        score = prediction[0, target]

        score.backward()

        fwd_hook.remove()
        bwd_hook.remove()

        gradient_agg = np.mean(self.gradients[0], axis=(1, 2))
        weighted_activations = np.sum(
            self.activations[0] * gradient_agg[:, np.newaxis, np.newaxis], axis=0
        )

        # only the activations that contribute positively
        contributing_weighted_activations = np.maximum(weighted_activations, 0)
        print(
            contributing_weighted_activations.max(),
            contributing_weighted_activations.min(),
        )

        # since the gradcam weighted activations are really small
        # then rounding, - as CV2 needs Uint8 to overlay the
        # heatmap on the image - to get in the 255 range from 0-1 tensors would round all the values to 0
        # so we can normalize the weights to 0-1 so that they are properly converted
        # and so the heatmap is accurate
        contributing_weighted_activations = (
            contributing_weighted_activations
            - np.min(contributing_weighted_activations)
        ) / (
            np.max(contributing_weighted_activations)
            - np.min(contributing_weighted_activations)
        )

        print(
            contributing_weighted_activations.max(),
            contributing_weighted_activations.min(),
        )

        upsampled_heatmap = cv2.resize(
            contributing_weighted_activations,
            (image.size(3), image.size(2)),
            interpolation=cv2.INTER_CUBIC,
        )

        probabilities = torch.sigmoid(prediction[0])
        _, top_category_id = torch.topk(
            probabilities, 1
        )  # index of the category the model is most confident about
        best_class_idx = top_category_id[0].item()
        best_class = LABELS[best_class_idx]
        self.save_heatmap(
            upsampled_heatmap, self.image_path, self.model_name, best_class
        )

        return

    def save_activations(self, module, input, output):
        self.activations.append(output.detach().cpu().numpy().squeeze())
        return

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients.append(grad_out[0].cpu().numpy().squeeze())
        return

    def save_heatmap(self, heatmap, image_path, model_name, best_class):
        heatmap_folder = Path("heatmaps/") / model_name
        heatmap_folder.mkdir(exist_ok=True, parents=True)
        heatmap_save_path = heatmap_folder / f"heatmap_top_prediction_{best_class}.png"
        plt.clf()

        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))

        heatmap = (heatmap * 255).round().astype(np.uint8)
        print(heatmap.max(), heatmap.min())
        heatmap = heatmap.squeeze()
        heatmap_coloured = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        print(image.shape, heatmap_coloured.shape)
        heatmap_on_img = cv2.addWeighted(heatmap_coloured, 0.5, image, 0.5, 0)

        cv2.imwrite(heatmap_save_path, heatmap_on_img)
        print(f"Successfully saved heatmap to {heatmap_save_path}")
        return


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc_model_path = input(
    "Enter the path to the pt file of the model you would like to evaluate with GradCam \n"
)
image_path = input("Enter the path to the image you would like to run grad-cam on \n")

gc_model = torch.load(gc_model_path, weights_only=False)
# get the model name
# adapted from - https://stackoverflow.com/a/78066321
# Posted by Jatinder Kumar
# Retrieved 2026-04-05, License - CC BY-SA 4.0
model_name = os.path.basename(os.path.dirname(gc_model_path))

with open("norm_const.json", "r") as f:
    mean_std_data = json.load(f)
mean = mean_std_data["mean"]
standard_deviation = mean_std_data["std"]

gradcam = GradCam(gc_model, model_name, image_path, mean, standard_deviation)
gradcam.compute_gradcam_heatmap()
