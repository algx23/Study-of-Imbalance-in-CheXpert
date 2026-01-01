import pandas as pd
import os
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from constants import LABELS

df = pd.read_csv("../train.csv")


# replace the blank columns for each disease with 0s
df[LABELS] = df[LABELS].fillna(0)
# replace -1 uncertain values with 0 - negative
df[LABELS] = df[LABELS].replace(-1, 0)

# drop lateral images for simplicity - in future experiments will compare all views
df = df.drop(df[df["Frontal/Lateral"] == "Lateral"].index)

# save to an excel file to test
df.to_csv("prepared data.csv", index=False)

# see the % of each class in the dataset - before sampling down to 10000
before_sample_mean = df[LABELS].mean(axis=0) * 100
print(f"Class Distribution after label cleaning - full dataset: \n{before_sample_mean}")

# get a sample of 10k

paths_to_images = df["Path"]
split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=len(paths_to_images.values) - 10000, random_state=0)
# test size is the number of images in the cleaned dataset, - 10,000 so that when MultilabelStratifiedShuffleSplit splits the data, i get 10,000 training images, which is all that is needed, and the rest are set as test images (though with test.csv provided this can be ignored)

# i am only subsetting the train.csv so the test indexes don't really matter
for train_index, test_index in split.split(paths_to_images, df[LABELS].values):
    X_train = paths_to_images.iloc[train_index] # stores the paths to the images - essentially the training images
    # X_test = paths_to_images.iloc[test_index]
    Y_train = df[LABELS].values[train_index] # stores the label values for each image, so x_train and y_train will line up
    # Y_test = df[LABELS].values[test_index]

#print(f"Overall Dataset Length: {len(paths_to_images)}")
#print(f"Number of || Training Images: {len(X_train)}, Training Labels: {len(Y_train)}")
#print(f"Number of || Testing Images: {len(X_test)}, Testing Labels: {len(Y_test)}")

#print(type(X_train), type(Y_train))

# when subsett-ing the data, there might be gaps in the row indexes in the numpy array
# eg, one example might be from row 2222, but the next example taken as part of
# the subset may be with index 2500
# this caused some of the labels to be blank, as when pandas tries to match an image
# to its labels, there is no corresponding label index for 2500 so pandas just makes it
# blank
# so i reset the indexes, so they both line up starting from 0
# source: https://www.datacamp.com/tutorial/pandas-reset-index-tutorial
subset_df = X_train.to_frame().reset_index(drop=True) # drop so the old indexes arent a column
subset_df[LABELS] = pd.DataFrame(Y_train, columns=LABELS)

subset_df.to_csv("subset.csv", index=False)

# class distribution of the subset
subset_mean = subset_df[LABELS].mean(axis=0) * 100
print(f"class distribution after sampling: \n{subset_mean}")

# splitting into validatation set for early stopping - 90/10
paths_to_training_images = subset_df['Path']
validation_split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=0)

for train_index, test_index in validation_split.split(paths_to_training_images, subset_df[LABELS].values):

    X_train = paths_to_training_images.iloc[train_index] # stores the paths to the images - essentially the training images
    X_train_val = paths_to_training_images.iloc[test_index]
    Y_train = subset_df[LABELS].values[train_index] # stores the label values for each image, so x_train and y_train will line up
    Y_train_val = subset_df[LABELS].values[test_index] 

training_df = X_train.to_frame().reset_index(drop=True)
training_df[LABELS] = pd.DataFrame(Y_train, columns=LABELS)

validation_df = X_train_val.to_frame().reset_index(drop=True)
validation_df[LABELS] = pd.DataFrame(Y_train_val, columns=LABELS)

training_df.to_csv("train.csv", index=False)
validation_df.to_csv("validation.csv", index=False)


validation_mean = validation_df[LABELS].mean(axis=0) * 100
print(f"class distribution after sampling: \n{round(validation_mean, 2)}")
