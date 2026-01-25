from argparse import ArgumentParser
from torchvision.transforms import (RandomRotation,
                                    RandomHorizontalFlip,
                                    RandomVerticalFlip,
                                    ColorJitter)


def parse_arguments():
    augments_for_experiment = []

    parser = ArgumentParser()
    parser.add_argument("--name", required=True, type=str)

    # to specify if there is any augment
    parser.add_argument("--augment", "--aug", action="store_true")

    # to specify if there is a class weight or not
    parser.add_argument("--class_weights", action="store_true")

    # a flag for each augmentation
    # Rotation, Horizontal Flip, Vertical Flip, Colour Jitter, CLAHE
    parser.add_argument("--rotate", type=int, choices= [0, 5, 10, 15, 30, 45], default=0)
    parser.add_argument("--vflip", type=float)
    parser.add_argument("--hflip", type=float)
    parser.add_argument("--jitter", type=float, nargs=2)
    parser.add_argument("--clahe", action="store_true")

    use_weights = False
    args = parser.parse_args()

    if args.vflip is not None and (args.vflip < 0 or args.vflip > 1):
        raise ValueError("vflip argument must be a floating point number between 0, and 1")
    if args.hflip is not None and (args.hflip < 0 or args.hflip > 1):
        raise ValueError("hflip argument must be a floating point number between 0, and 1")

    if args.class_weights:
        use_weights = True
    print(args)


    if args.rotate:
        augments_for_experiment.append(RandomRotation(args.rotate))
    if args.vflip:
        augments_for_experiment.append(RandomVerticalFlip(args.vflip))
    if args.hflip:
        augments_for_experiment.append(RandomHorizontalFlip(args.hflip))
    if args.jitter:
        # since cheXpert images are grayscale, they do not have a hue or a saturation
        augments_for_experiment.append(ColorJitter(brightness=args.jitter[0], contrast=args.jitter[1]))
    if args.clahe: # TODO: CLAHE implementation with openCV
        NotImplemented

    return (args.name, augments_for_experiment, use_weights)

