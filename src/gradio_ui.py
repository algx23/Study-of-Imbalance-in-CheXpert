import json
import os
from pathlib import Path

import gradio as gr


from pathlib import Path
import torch
from torchvision.transforms import Resize, Normalize, Compose, ToTensor
import cv2
from PIL import Image

from gradcam import GradCam
from model import BaselineModel
import predict_single_image


norm_const_file = "norm_const.json"
if os.path.exists(norm_const_file):
    with open(norm_const_file, "r") as f:
        mean_std_data = json.load(f)
        MEAN = mean_std_data["mean"]
        STD = mean_std_data["std"]
else:
    MEAN, STD = 0.5064598321914673, 0.2867761871433037


def get_threshold_for_inference(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold_file = Path(model_path).parent / "thresholds.json"
    if os.path.exists(threshold_file):
        with open(threshold_file, "r", encoding="utf-8") as threshold_file:
            data = json.load(threshold_file)
            threshold = data["thresholds"]
    else:
        threshold = [0.3] * 13
    return threshold


def select_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(".")

    models_available = []
    # https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python
    for file in project_root.rglob("*.pt"):
        try:
            model = torch.load(file, weights_only=False, map_location=device)

            if "tensor" not in file.parent.name and isinstance(model, BaselineModel):
                models_available.append(str(file.relative_to(project_root)))
        except Exception as e:
            print(f"ERROR: {e}")

    print(models_available)
    return models_available


def run_heatmap_analysis(model_path, image_path, mean=MEAN, std=STD):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f"{model_path}", weights_only=False, map_location=device)
    model_name = Path(model_path).parent.name
    gradcam = GradCam(model, model_name, image_path, mean=mean, std=std)
    heatmap_path = gradcam.compute_gradcam_heatmap()
    print(heatmap_path)
    return Image.open(heatmap_path)


def classify_single_image(model_path, image_path, mean=MEAN, std=STD):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = predict_single_image.prepare_image_for_classification(image_path)
    model = torch.load(f"{model_path}", weights_only=False, map_location=device)

    threshold = get_threshold_for_inference(model_path)
    print(f"THRESHOLD: {threshold}")

    classification_result = predict_single_image.make_classification(
        model, image, threshold=threshold, device=device
    )
    return str(classification_result)


with gr.Blocks(title="Predict X-Ray Classification", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
  Provide a model, and the X-Ray in order to generate predictions, and a GRAD-CAM heatmap
  """
    )

    model_path = gr.Dropdown(
        choices=select_model(), label="Available Models to Evaluate", interactive=True
    )

    gr.Markdown("Select an image to evaluate")
    with gr.Row():
        original_image = gr.Image(label="Original Image", type="filepath")
        output_gradcam_image = gr.Image(
            type="pil", label=f"GradCam Result", width=224, height=224
        )
        gradcam_button = gr.Button(value="Generate Gradcam Heatmap")

    gradcam_button.click(
        fn=run_heatmap_analysis,
        inputs=[model_path, original_image],
        outputs=output_gradcam_image,
    )

    with gr.Row():
        inference_result = gr.Textbox(label="Classification Result")
        inference_button = gr.Button(value="Classify Image")

    inference_button.click(
        fn=classify_single_image,
        inputs=[model_path, original_image],
        outputs=inference_result,
    )


demo.launch(allowed_paths=["./"])
