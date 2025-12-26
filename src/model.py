import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class BaselineModel(nn.Module):
    """
    The baseline Model for my experiments. The model has 2 convolution layers,
    with 2x2 filters, with 32 filters for the first layer, and 64 filters for the second
    convolution layer respectively.

    After each convolution layer, ReLu Activation is used to preserve some complexity
    in the model, and max pooling is used to reduce the spatial dimensions of the image
    which can help computation efficiency, followed by a fully connected layer


    Reference:
    -https://www.datacamp.com/tutorial/pytorch-cnn-tutorial?utm_cid=23340058065&utm_aid=192632748929&utm_campaign=230119_1-ps-other~dsa-tofu~python_2-b2c_3-emea_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&utm_loc=9046182-&utm_mtd=-c&utm_kw=&utm_source=google&utm_medium=paid_search&utm_content=ps-other~emea-en~dsa~tofu~tutorial~python&gad_source=1&gad_campaignid=23340058065&gbraid=0AAAAADQ9WsEnPx1QuzTf75-w9DiI2F56n&gclid=Cj0KCQiAgbnKBhDgARIsAGCDdlfBNNqu8bDrwzSi6qoHlb7BvVRpXN2uK5Y-zLXefV6JPnGCLD1PEr8aAovREALw_wcB
    - https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    - https://dingyan89.medium.com/calculating-parameters-of-convolutional-and-fully-connected-layers-with-keras-186590df36c6
    """
    def __init__(self):
        super().__init__()
        # convolution layer 1: 3x3 filters, 32 filters
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2) 
        self.conv_2 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=3, stride=2)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        # fully connected layer
    
        self.fully_connected = nn.Linear(10816, 14 ) # 10,000 images, that have been flattened?


    def forward(self, images):
        # images into conv1
       
        # input: 224x224, output shape = 224-3/2 + 1 = 111.0 with floor division
        images = self.conv_1(images)
        # relu 1
        images = F.relu(images)
        # max pool 1

        # input: 111x111 from conv1
        images = self.max_pool_1(images)
        # output: 111-2+2*0 / 2 + 1 = 55

        # conv2 -> input shape = 55x55
        images = self.conv_2(images)
        # output = 55-3/2 + 1 = 27x27 images
        # relu 2
        images = F.relu(images)
        #max pool 2 -> input image = 27x27
        images = self.max_pool_2(images)
        #output = 27-2*2(0)/2 + 1 =13x13 image

        # flatten
        # input = all 13x13 output rfom the 64 filters
        images = self.flatten(images)
        # output = 13 x 13 x 64filters = 10816
        images = self.fully_connected(images)

        return images
        


