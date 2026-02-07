import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import decode_image
from constants.control_variables import LABELS
from torch import float32
import cv2
from PIL import Image

class ChexpertDataset(Dataset):
    """
    Custom dataset for the ChexPert dataset.
    This will be loaded into the DataLoader class

    Stuff used to help:
    - PyTorch Dataset Documentation: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    - PyTorch DataLoader Documentation: https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    - PyTorch decode_image Documentation to see what it did: https://docs.pytorch.org/vision/stable/generated/torchvision.io.decode_image.html#torchvision.io.decode_image
    """
    def __init__(self, subset_csv_dir, image_dir, transform=None, target_transform=None, use_clahe=False):
        self.df = pd.read_csv(subset_csv_dir)
        self.labels = self.df[LABELS] # all the columns with labels
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.use_clahe=use_clahe

    def __len__(self):
        """Returns the number of images in the dataset

        Returns: int: number of images in the dataset
        """
        return len(self.labels)

    def __getitem__(self, idx): # return an image and its labels for a specific index / row number
        path_to_single_image = os.path.join(self.image_dir, self.df.loc[idx, "Path"])
        image = cv2.imread(path_to_single_image, cv2.IMREAD_GRAYSCALE)

        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)

        image = Image.fromarray(image)

        label = self.labels.iloc[idx] # get all the label values for each image - each image is in its own row
        # normalize image pixel values to between 0 and 1 initially,
        # divide by 255 -> could use ToTensor, but decode_image already outputs a tensor
        # so in the transform id need to do Tensor -> Numpy array for ToTensor -> back to Tensor which seems inefficient
        # image = image / 255 
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        #image = image.to(dtype=float32) # got an error so fixed

        label = label.to_numpy() # to address a error i got saying pytorch expects a numpy array rather than a pandas series
        return image, label
    
