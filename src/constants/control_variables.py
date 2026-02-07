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

MODEL_NAME, *VARS_FOR_EXPERIMENT = parse_arguments()

NUM_EPOCHS = 1 # for testing
CHEXPERT_COMP_LABELS = ["Atelectasis",
                        "Cardiomegaly",
                        "Consolidation",
                        "Edema",
                        "Pleural Effusion"]
