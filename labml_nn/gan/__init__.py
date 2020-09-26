import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data

from labml_helpers.module import Module


def create_labels(n: int, r1: float, r2: float, device: torch.device = None):
    return torch.empty(n, 1, requires_grad=False, device=device).uniform_(r1, r2)


class DiscriminatorLogitsLoss(Module):
    def __init__(self, smoothing: float = 0.2):
        super().__init__()
        self.loss_true = nn.BCEWithLogitsLoss()
        self.loss_false = nn.BCEWithLogitsLoss()
        self.smoothing = smoothing
        self.register_buffer('labels_true', create_labels(256, 1.0 - smoothing, 1.0), False)
        self.register_buffer('labels_false', create_labels(256, 0.0, smoothing), False)

    def __call__(self, logits_true: torch.Tensor, logits_false: torch.Tensor):
        if len(logits_true) > len(self.labels_true):
            self.register_buffer("labels_true",
                                 create_labels(len(logits_true), 1.0 - self.smoothing, 1.0, logits_true.device), False)
        if len(logits_false) > len(self.labels_false):
            self.register_buffer("labels_false",
                                 create_labels(len(logits_false), 0.0, self.smoothing, logits_false.device), False)

        return self.loss_true(logits_true, self.labels_true[:len(logits_true)]), \
               self.loss_false(logits_false, self.labels_false[:len(logits_false)])


class GeneratorLogitsLoss(Module):
    def __init__(self, smoothing: float = 0.2):
        super().__init__()
        self.loss_true = nn.BCEWithLogitsLoss()
        self.smoothing = smoothing
        self.register_buffer('fake_labels', create_labels(256, 1.0 - smoothing, 1.0), False)

    def __call__(self, logits: torch.Tensor):
        if len(logits) > len(self.fake_labels):
            self.register_buffer("fake_labels",
                                 create_labels(len(logits), 1.0 - self.smoothing, 1.0, logits.device), False)

        return self.loss_true(logits, self.fake_labels[:len(logits)])
