import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd

folder_to_results = input("Enter the path to the results folder containing hte experiments whose loss graph you want to fix: \n")

for experiment in os.listdir(folder_to_results):
    plt.clf()
    loss_pr_auc_df_path = Path(folder_to_results) /experiment / "train" / "avg_epoch_loss.csv"
    df = pd.read_csv(loss_pr_auc_df_path)
    epochs, training_losses, validation_prauc = df["Epoch"], df["Train Loss"], df["Validation Loss"]

    plt.plot(epochs, training_losses, label="Training Loss")
    plt.plot(epochs, validation_prauc, label="Validation PR-AUC")
    plt.ylabel("Value")
    plt.xlabel("Number of epochs completed")
    plt.legend()
    plt.title("Training Loss and Validation PR-AUC over epochs")
    save_path = Path(folder_to_results) / experiment / f"{experiment}_FIXED_loss_prauc_graph.png"
    plt.savefig(save_path)
    plt.close()


