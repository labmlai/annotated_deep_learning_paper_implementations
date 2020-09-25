import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data

from labml_helpers.module import Module


class DiscriminatorLogitsLoss(Module):
    def __init__(self):
        super().__init__()
        self.loss_true = nn.BCEWithLogitsLoss()
        self.loss_false = nn.BCEWithLogitsLoss()
        self.register_buffer('labels_true', torch.ones(256, 1, requires_grad=False), False)
        self.register_buffer('labels_false', torch.ones(256, 1, requires_grad=False), False)

    def __call__(self, logits_true: torch.Tensor, logits_false: torch.Tensor):
        if len(logits_true) > len(self.labels_true):
            self.register_buffer("labels_true",
                                 self.labels_true.new_ones(len(logits_true), 1, requires_grad=False), False)
        if len(logits_false) > len(self.labels_false):
            self.register_buffer("labels_false",
                                 self.labels_false.new_ones(len(logits_false), 1, requires_grad=False), False)

        loss = (self.loss_true(logits_true, self.labels_true[:len(logits_true)]) +
                self.loss_false(logits_false, self.labels_false[:len(logits_false)]))

        return loss


class GeneratorLogitsLoss(Module):
    def __init__(self):
        super().__init__()
        self.loss_true = nn.BCEWithLogitsLoss()
        self.register_buffer('fake_labels', torch.ones(256, 1, requires_grad=False), False)

    def __call__(self, logits: torch.Tensor):
        if len(logits) > len(self.fake_labels):
            self.register_buffer("fake_labels",
                                 self.fake_labels.new_ones(len(logits), 1, requires_grad=False), False)

        return self.loss_true(logits, self.fake_labels[:len(logits)])
