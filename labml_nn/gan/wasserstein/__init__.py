import torch
import torch.utils.data
from torch.nn import functional as F

from labml_helpers.module import Module


class DiscriminatorLoss(Module):
    """
    ## Discriminator Loss
    """

    def __init__(self):
        super().__init__()

    def __call__(self, logits_true: torch.Tensor, logits_false: torch.Tensor):
        """
        `logits_true` are logits from $D(\pmb{x}^{(i)})$ and
        `logits_false` are logits from $D(G(\pmb{z}^{(i)}))$
        """

        return F.relu(1 - logits_true).mean(), F.relu(1 + logits_false).mean()


class GeneratorLoss(Module):
    """
    ## Generator Loss
    """

    def __init__(self):
        super().__init__()

    def __call__(self, logits: torch.Tensor):
        return -logits.mean()


def _create_labels(n: int, r1: float, r2: float, device: torch.device = None):
    """
    Create smoothed labels
    """
    return torch.empty(n, 1, requires_grad=False, device=device).uniform_(r1, r2)
