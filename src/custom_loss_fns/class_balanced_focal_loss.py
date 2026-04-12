from torch import Tensor
import torch
from torch.nn import BCEWithLogitsLoss
import torch.nn as nn


class ClassBalancedFocalLoss(nn.Module):
    """
    A custom implementation of the Class Balanced Focal Loss (CBFL)
    Rather than using an alpha_t weighting factor, like the original
    focal loss, CBFL uses a parameter Beta, which acts similarly
    to the alpha parameter, but is defined by:

    (1 - beta) / (1-beta^n_y) where n_y is the number of samples in
    the ground truth label y -> i.e when y=1, the weight is calculated
    with the number of positive samples, and vice versa when y=0

    Additionally, these "Weights" are normalized such that
    \\Sum alpha_i = N_c where N_c is the number of classes.
    References:
        - Original Focal Loss Paper: https://arxiv.org/pdf/1708.02002
        - Class Balanced Focal Loss: https://arxiv.org/pdf/1901.05555
    """

    def __init__(self, beta, gamma):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.bce_loss = BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: Tensor, true_labels: Tensor):

        p = torch.sigmoid(logits)
        ce = self.bce_loss(logits, true_labels)

        balanced_focal_loss = ce * (
            self.beta[:, 0] * true_labels * (1 - p) ** self.gamma
            + self.beta[:, 1] * (1 - true_labels) * p**self.gamma
        )

        return torch.mean(balanced_focal_loss)
