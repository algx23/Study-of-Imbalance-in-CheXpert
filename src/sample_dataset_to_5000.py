import pandas as pd

df = pd.read_csv("../train.csv")
LABELS = ["No Finding",
          "Enlarged Cardiomediastinum",
          "Cardiomegaly",
          "Lung Opacity",
          "Lung Lesion",
          "Edema",
          "Consolidation",
          "Pneumonia",
          "Atelectasis",
          "Pneumothorax",
          "Pleural Effusion",
          "Pleural Other",
          "Fracture"
          ]

# replace the blank columns for each disease with 0s
df[LABELS] = df[LABELS].fillna(0)
# replace -1 uncertain values with 0 - negative
df[LABELS] = df[LABELS].replace(-1, 0)

# save to an excel file to test
df.to_csv("prepared data.csv", index=False)
