"""
---
title: 2D Convolution Layer with Weight Standardization
summary: >
 A PyTorch implementation/tutorial of a 2D Convolution Layer with Weight Standardization.
---

# 2D Convolution Layer with Weight Standardization

This is an implementation of a 2 dimensional convolution layer with [Weight Standardization](./index.html)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from labml_nn.normalization.weight_standardization import weight_standardization


class Conv2d(nn.Conv2d):
    """
    ## 2D Convolution Layer

    This extends the standard 2D Convolution layer and standardize the weights before the convolution step.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 eps: float = 1e-5):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=groups,
                                     bias=bias,
                                     padding_mode=padding_mode)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, weight_standardization(self.weight, self.eps), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def _test():
    """
    A simple test to verify the tensor sizes
    """
    conv2d = Conv2d(10, 20, 5)
    from labml.logger import inspect
    inspect(conv2d.weight)
    import torch
    inspect(conv2d(torch.zeros(10, 10, 100, 100)))


if __name__ == '__main__':
    _test()
