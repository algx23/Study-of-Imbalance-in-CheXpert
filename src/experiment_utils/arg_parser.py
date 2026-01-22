from argparse import ArgumentParser
from torchvision.transforms import (RandomRotation,
                                    RandomHorizontalFlip,
                                    RandomVerticalFlip,
                                    ColorJitter)


def parse_arguments():
    augments_for_experiment = []

    parser = ArgumentParser()
    # to specify if there is any augment
    parser.add_argument("--augment", "--aug", action="store_true")

    # to specify if there is a class weight or not
    parser.add_argument("--class_weights", action="store_true")

    # a flag for each augmentation
    # Rotation, Horizontal Flip, Vertical Flip, Colour Jitter, CLAHE
    parser.add_argument("--rotate", type=int, choices= [0, 5, 10, 15, 30, 45], default=5)
    parser.add_argument("--vflip", type=float)
    parser.add_argument("--hflip", type=float)
    parser.add_argument("--jitter", action="store_true")
    parser.add_argument("--clahe", action="store_true")

    args = parser.parse_args()

    if args.vflip is not None and (args.vflip < 0 or args.vflip > 1):
        raise ValueError("vflip argument must be a floating point number between 0, and 1")
    
    if args.hflip is not None and (args.hflip < 0 or args.hflip > 1):
        raise ValueError("hflip argument must be a floating point number between 0, and 1")

    print(args)

    if args.rotate:
        augments_for_experiment.append(RandomRotation(args.rotate))
    if args.vflip:
        augments_for_experiment.append(RandomVerticalFlip(args.vflip))
    if args.hflip:
        augments_for_experiment.append(RandomHorizontalFlip(args.hflip))
    if args.jitter:
        augments_for_experiment.append(ColorJitter())
    if args.clahe: # TODO: CLAHE implementation with openCV
        pass

    print(augments_for_experiment)

    return augments_for_experiment

