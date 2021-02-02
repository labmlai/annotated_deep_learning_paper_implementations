"""
---
title: Label Smoothing Loss
summary: >
  This is an implementation of label smoothing loss, that can be used as
  an alternative to cross entropy loss for improved accuracy.
---

# Label Smoothing Loss
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from labml_helpers.module import Module


class LabelSmoothingLoss(Module):
    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        assert x.shape[1] == self.size
        true_dist = x.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.loss(x, true_dist.detach())


def _test_label_smoothing():
    smooth_loss = LabelSmoothingLoss(5, 0, 0.4)
    predict = torch.tensor([[0, 0.2, 0.7, 0.1, 0],
                            [0, 0.2, 0.7, 0.1, 0],
                            [0, 0.2, 0.7, 0.1, 0]], dtype=torch.float)
    _ = smooth_loss(predict.log(),
                    torch.tensor([2, 1, 0], dtype=torch.long))

    # Show the target distributions expected by the system.
    plt.imshow(smooth_loss.true_dist)
    plt.show()

    smooth_loss = LabelSmoothingLoss(5, 0, 0.1)

    def loss_sample(x):
        d = x + 3 * 1
        predict2 = torch.tensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                 ], dtype=torch.float)
        # print(predict)
        return smooth_loss(predict2.log(),
                           torch.tensor([1], dtype=torch.long)).item()

    plt.plot(np.arange(1, 100), [loss_sample(x) for x in range(1, 100)])
    plt.show()


if __name__ == '__main__':
    _test_label_smoothing()
