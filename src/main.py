from chexpert_dataset import ChexpertDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Resize # to resize all images
from utils import calculate_mean_and_standard_deviation

resize_transform = Resize((224, 224)) # some images are different sizes so resize them all to the same size
dataset = ChexpertDataset("subset.csv", "D:/dataset fyp/", transform=resize_transform)


print(f"dataset size: {dataset.__len__()}")

# testing with just 1 image for now to test logic
image, labels = dataset[0]
#print(f"label type: {type(labels)}")

#print(f"image shape: {image.shape}\n image : {image}\n") # grayscale(1 channel) 224x224 image
#print(f"labels: {labels.shape} || {labels}\n") # checking the labels exist properly for the image

dataset_loader = DataLoader(dataset, batch_size=25, shuffle=True)
mean, standard_deviation = calculate_mean_and_standard_deviation(dataset_loader)
print(f"mean = {mean}, standard deviation = {standard_deviation}")


# checking the data is loaded - from PyTorch DataLoader Documentation
#train_features, train_labels = next(iter(dataset_loader))
#print(f"Feature shape: {train_features.size()} ")
#print(f"label batch shape: {train_labels.size()}")
