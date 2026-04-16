from argparse import ArgumentParser
from torchvision.transforms import (
    RandomRotation,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
)


def parse_arguments():
    augments_for_experiment = []

    parser = ArgumentParser()
    help_strings = {
        "experiment root": "Specifies the parent folder of the train | evaluation pipeline to be executed",
        "name": "Specifies the name of the model. All figures, and metrics generated will use this as the file name",
        "class_weights": "if included class weights are calculated and used in Weighted BCE Loss",
        "rotate": "the angle x to rotate images by +- x degrees",
        "hflip": "horizontal flip: --hflip p : where p is the probability of an image being flipped",
        "vflip": "vertical flip: --vflip p : where p is the probability of an image being flipped",
        "jitter": "color jitter: adjust brightness and contrast by amount provided",
        "clahe": "if included, use Contrast Limited Histogram Equalization",
        "dropout": "if included, dropout layers will be added to the model after the ReLu Activation function is applied",
        "bn": "if included Batch Normalization will be added to the model",
        "focal_loss": "Set the loss function to focal loss with the provided gamma value",
        "balanced_focal_loss": "if inlcuded, class balanced focal loss will be used",
        "mixup": "if included MixUp augmentation will be used",
        "threshold": "Fixed: use a 0.3 fixed threshold for every model. Optimal: Calculate each model's optimal class thresholds based on maximizing the F1 Score",
    }

    # experiment name -> results saved to eg experiment-root/model1 experiment-root/model2
    parser.add_argument(
        "--experiment_root",
        type=str,
        help=help_strings["experiment root"],
        default="results",
    )

    # model name / experiment name - all metrics will be saved in a folder under this name
    parser.add_argument(
        "--name", required=False, type=str, help=help_strings["name"], default="NA"
    )

    # to specify if there is a class weight or not
    parser.add_argument(
        "--class_weights", action="store_true", help=help_strings["class_weights"]
    )

    # a flag for each augmentation
    # Rotation, Horizontal Flip, Vertical Flip, Colour Jitter, CLAHE
    parser.add_argument(
        "--rotate",
        type=int,
        choices=[0, 5, 10, 45],
        default=0,
        help=help_strings["rotate"],
    )
    parser.add_argument("--vflip", type=float, help=help_strings["vflip"])
    parser.add_argument("--hflip", type=float, help=help_strings["hflip"])
    parser.add_argument("--jitter", type=float, nargs=2, help=help_strings["jitter"])
    parser.add_argument("--clahe", action="store_true", help=help_strings["clahe"])
    parser.add_argument("--dropout", action="store_true", help=help_strings["dropout"])
    parser.add_argument("--bn", action="store_true", help=help_strings["bn"])
    parser.add_argument(
        "--focal_loss",
        type=float,
        required=False,
        choices=[2, 0.5, 1],
        help=help_strings["focal_loss"],
    )
    parser.add_argument(
        "--balanced_focal_loss",
        action="store_true",
        help=help_strings["balanced_focal_loss"],
    )
    parser.add_argument("--mixup", action="store_true", help=help_strings["mixup"])
    parser.add_argument(
        "--threshold",
        type=str,
        choices=["fixed", "optimal"],
        default="fixed",
        help=help_strings["threshold"],
    )

    (
        use_weights,
        use_clahe,
        use_dropout,
        use_batch_norm,
        use_cbfl,
        focal_loss_gamma,
        use_mixup,
    ) = (False, False, False, False, False, None, False)
    args = parser.parse_args()

    if args.vflip is not None and (args.vflip < 0 or args.vflip > 1):
        raise ValueError(
            "vflip argument must be a floating point number between 0, and 1"
        )
    if args.hflip is not None and (args.hflip < 0 or args.hflip > 1):
        raise ValueError(
            "hflip argument must be a floating point number between 0, and 1"
        )

    if args.class_weights:
        use_weights = True

    if args.rotate:
        augments_for_experiment.append(RandomRotation(args.rotate))
    if args.vflip:
        augments_for_experiment.append(RandomVerticalFlip(args.vflip))
    if args.hflip:
        augments_for_experiment.append(RandomHorizontalFlip(args.hflip))
    if args.jitter:
        # since cheXpert images are grayscale, they do not have a hue or a saturation
        augments_for_experiment.append(
            ColorJitter(brightness=args.jitter[0], contrast=args.jitter[1])
        )

    if args.clahe:
        use_clahe = True
    if args.dropout:
        use_dropout = True
    if args.bn:
        use_batch_norm = True
    if args.focal_loss is not None:
        focal_loss_gamma = args.focal_loss
    if args.balanced_focal_loss:
        use_cbfl = True

    if args.mixup:
        use_mixup = True

    threshold = args.threshold

    print(args)
    return (
        args.experiment_root,
        args.name,
        augments_for_experiment,
        use_weights,
        use_clahe,
        use_dropout,
        use_batch_norm,
        focal_loss_gamma,
        use_cbfl,
        use_mixup,
        threshold,
    )
