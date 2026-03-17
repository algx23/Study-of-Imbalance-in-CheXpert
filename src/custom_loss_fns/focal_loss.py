from torch import Tensor
import torch
from torch.nn import BCEWithLogitsLoss
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    A custom Focal Loss Implementation for multi-label classification

    References:
    - Focal Loss for Dense Object Detection [https://arxiv.org/pdf/1708.02002]
    - the idea for indicator functions: https://math.stackexchange.com/questions/5101858/how-do-i-write-a-piecewise-function-in-a-single-equation
    - Indicator Functions (to turn the focal loss into one equation): [https://en.wikipedia.org/wiki/Indicator_function]
    """

    def __init__(self, alpha: Tensor, gamma: float) -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = BCEWithLogitsLoss(
            reduction="none"
        )  # needed to calculate the log(p) part of the equation
        return

    def forward(self, logits: Tensor, true_labels: Tensor) -> Tensor:
        """
        Calculates the focal loss of a given batch
        p_t = p if y=1
        p_t = 1-p if y=0
        FL(x) = (-alpha(1-p_t)^gamma) * log(p_t)

        See Dissertation for full mathematical derviation -> TODO: Add to my dissertation

        Args:
        - batch: Tensor: the batch of images of which to calculate the loss
        """

        p = torch.sigmoid(logits)
        ce = self.bce_loss(logits, true_labels)
        # compute the loss across all classes and then return it as a single value
        # so that it fits with my existing training loop
        focal_loss = ce * (
            self.alpha * true_labels * (1 - p) ** self.gamma
            + (1 - self.alpha) * (1 - true_labels) * p**self.gamma
        )
        return torch.mean(focal_loss)
