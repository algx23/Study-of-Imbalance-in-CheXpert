import pandas as pd
import numpy as np
from constants import LABELS
from path_creator import EVAL_DATA_PATH, MATRIX_PATH
from sklearn.metrics import (classification_report,
                             multilabel_confusion_matrix,
                             ConfusionMatrixDisplay)
class MetricCalculator():

    def __init__(self, predictions, truth):
        self.predictions = predictions
        self.truth = truth
        return

    def create_classification_report(self):

       report = classification_report(y_true=self.truth, y_pred=self.predictions, target_names=LABELS, output_dict=True, zero_division=0)

       # save the report to a csv
       report_df = pd.DataFrame(report).transpose()

       report_path = EVAL_DATA_PATH / "classification_report.csv"
       report_df.to_csv(report_path) 
       print(report)
       return

    def create_per_class_confusion_matrices(self):
       confusion_matrices = multilabel_confusion_matrix(y_true=np.array(self.truth), y_pred=np.array(self.predictions))
       for i in range(len(LABELS)): # print the confusion matrix for the first class

           matrix_plot = ConfusionMatrixDisplay(confusion_matrices[i])
           matrix_plot.plot()

           # save the plots
           matrix_filepath = MATRIX_PATH / f"{LABELS[i]}_confusion_matrix.png"
           matrix_plot.figure_.savefig(matrix_filepath)

       return

    def calculate_metrics(self):
        self.create_classification_report()
        self.create_per_class_confusion_matrices()
        return

