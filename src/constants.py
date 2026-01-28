from experiment_utils.arg_parser import parse_arguments

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

NUM_EPOCHS = 3 # for testing
MODEL_NAME = parse_arguments()[0]
CHEXPERT_COMP_LABELS = ["Atelectasis",
                        "Cardiomegaly",
                        "Consolidation",
                        "Edema",
                        "Pleural Effusion"]
