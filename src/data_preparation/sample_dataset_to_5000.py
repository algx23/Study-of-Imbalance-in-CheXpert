import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


class DataSubsetter:
    """Class to create hte train, validation and test splits"""

    def __init__(
        self,
        ORIGINAL_DATASET_PATH,
        TRAIN_SET_PATH,
        VALIDATION_SET_PATH,
        TEST_SET_PATH,
        LABELS,
    ):
        """Initialize the subsetter class

        Args:
            ORIGINAL_DATASET_PATH (string): path to the initial csv from which the subset is made
            TRAIN_SET_PATH (string): path to which the train split will be saved
            VALIDATION_SET_PATH (string): path to which the validation split csv will be saved
            TEST_SET_PATH (string): path to which the test split csv will be saved
            LABELS (List[String]): list of Label names to be used
        """
        self.ORIGINAL_DATASET_PATH = ORIGINAL_DATASET_PATH
        self.TRAIN_SET_PATH = TRAIN_SET_PATH
        self.VALIDATION_SET_PATH = VALIDATION_SET_PATH
        self.TEST_SET_PATH = TEST_SET_PATH
        self.LABELS = LABELS

    def create_subset_train_validation(self):
        """Creates the train and validation splits, by first subsetting the full train csv into a
        10k subset and then splitting this subset into a train and validation split, in a 90:10 ratio
        """
        df = pd.read_csv(self.ORIGINAL_DATASET_PATH)

        # replace the blank columns for each disease with 0s
        df[self.LABELS] = df[self.LABELS].fillna(0)
        # replace -1 uncertain values with 0 - negative
        df[self.LABELS] = df[self.LABELS].replace(-1, 0)

        # drop lateral images for simplicity - in future experiments will compare all views
        df = df.drop(df[df["Frontal/Lateral"] == "Lateral"].index)

        # save to an excel file to test
        df.to_csv("prepared data.csv", index=False)

        # see the % of each class in the dataset - before sampling down to 10000
        before_sample_mean = df[self.LABELS].mean(axis=0) * 100
        print(
            f"Class Distribution after label cleaning - full dataset: \n{before_sample_mean}"
        )

        # get a sample of 10k
        paths_to_images = df["Path"]
        split = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=len(paths_to_images.values) - 10000, random_state=0
        )
        # len(paths_to_images) = ~224k -> test size = 214k~ so train index provides 10k values

        # i am only subsetting the train.csv so the test indexes don't really matter
        for train_index, test_index in split.split(
            paths_to_images, df[self.LABELS].values
        ):
            X_train = paths_to_images.iloc[
                train_index
            ]  # stores the paths to the images - essentially the training images
            Y_train = df[self.LABELS].values[
                train_index
            ]  # stores the label values for each image, so x_train and y_train will line up

        # when subsett-ing the data, there might be gaps in the row indexes in the numpy array
        # eg, one example might be from row 2222, but the next example taken as part of
        # the subset may be with index 2500
        # this caused some of the labels to be blank, as when pandas tries to match an image
        # to its labels, there is no corresponding label index for 2500 so pandas just makes it
        # blank
        # so i reset the indexes, so they both line up starting from 0
        # source: https://www.datacamp.com/tutorial/pandas-reset-index-tutorial
        subset_df = X_train.to_frame().reset_index(
            drop=True
        )  # drop so the old indexes arent a column
        subset_df[self.LABELS] = pd.DataFrame(Y_train, columns=self.LABELS)

        subset_df.to_csv("subset.csv", index=False)

        # class distribution of the subset
        subset_mean = subset_df[self.LABELS].mean() * 100
        print(f"class distribution after sampling: \n{subset_mean}")

        # splitting the subset into train/val - 90/10
        paths_to_training_images = subset_df["Path"]
        validation_split = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=1000, random_state=0
        )

        # train index will make up the train set, test indices make up the
        # validation set
        for train_index, test_index in validation_split.split(
            paths_to_training_images, subset_df[self.LABELS].values
        ):

            X_train = paths_to_training_images.iloc[
                train_index
            ]  # stores the paths to the images - essentially the training images
            X_train_val = paths_to_training_images.iloc[test_index]
            Y_train = subset_df[self.LABELS].values[
                train_index
            ]  # stores the label values for each image, so x_train and y_train will line up
            Y_train_val = subset_df[self.LABELS].values[test_index]

        training_df = X_train.to_frame().reset_index(drop=True)
        training_df[self.LABELS] = pd.DataFrame(Y_train, columns=self.LABELS)

        validation_df = X_train_val.to_frame().reset_index(drop=True)
        validation_df[self.LABELS] = pd.DataFrame(Y_train_val, columns=self.LABELS)

        training_df.to_csv(self.TRAIN_SET_PATH, index=False)
        validation_df.to_csv(self.VALIDATION_SET_PATH, index=False)

        validation_mean = validation_df[self.LABELS].mean() * 100
        print(
            f"class distribution after sampling validation: \n{round(validation_mean, 2)}"
        )

        return

    def create_test_data_csv(self):
        """Creates the Test data csv containing the rows of images and their labels
        Removes the subset from the original, full set, and then stratify samples a 2000
        image subset, to ensure no images from the train or validation set enter the test set
        """
        all_df = pd.read_csv("prepared data.csv")
        subset_df = pd.read_csv("subset.csv")

        # remove the rows that are in the train and validation
        # sets so that when splitting it none of the rows get in
        # either set - prevent leakage

        # adapted from: https://stackoverflow.com/questions/44706485/how-to-remove-rows-in-a-pandas-dataframe-if-the-same-row-exists-in-another-dataf
        df_fully_removed = (
            pd.merge(all_df, subset_df, indicator=True, how="outer")
            .query("_merge=='left_only'")
            .drop("_merge", axis=1)
        )

        # split this to get the test set
        paths_to_images = df_fully_removed["Path"]
        labels_for_each_row = df_fully_removed[self.LABELS].values

        test_split = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=2000, random_state=42
        )
        for _, test_index in test_split.split(paths_to_images, labels_for_each_row):
            X_test = paths_to_images.iloc[test_index]
            Y_test = labels_for_each_row[test_index]

        print(type(X_test))
        print(type(Y_test))
        print(X_test.shape)
        print(Y_test.shape)

        test_df = X_test.to_frame().reset_index(drop=True)
        test_df[self.LABELS] = Y_test
        print("Test set distribution: \n")
        print(test_df[self.LABELS].mean() * 100)
        test_df.to_csv(self.TEST_SET_PATH)

        return

    def calculate_patient_overlap(self):
        train_set = pd.read_csv(self.TRAIN_SET_PATH)
        validation_set = pd.read_csv(self.VALIDATION_SET_PATH)
        test_set = pd.read_csv(self.TEST_SET_PATH)

        train_patient_ids = set(train_set["Path"].str.split("/").str[2])
        val_patient_ids = set(validation_set["Path"].str.split("/").str[2])
        test_patient_ids = set(test_set["Path"].str.split("/").str[2])

        # % of patients in train-val
        pct_train_val_overlap = len(train_patient_ids & val_patient_ids) /  len(val_patient_ids)
        # % of patients that are in test that were seen in train or val
        ids_in_train_and_val = train_patient_ids | val_patient_ids
        pct_all_test_overlap = len(ids_in_train_and_val & test_patient_ids) /  len(test_patient_ids)

        print(f"% of images that were in validation that are also in train: {pct_train_val_overlap * 100}")
        print(f"% of images that were in either train or val that are also in test: {pct_all_test_overlap * 100}")
        return

