import torch
import torch.nn as nn
from torch.nn import functional as F

from labml_nn.normalization.weight_standardization import weight_standardization


class Conv2d(nn.Conv2d):
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
    conv2d = Conv2d(10, 20, 5)
    from labml.logger import inspect
    inspect(conv2d.weight)
    import torch
    conv2d(torch.zeros(10, 10, 100, 100))


if __name__ == '__main__':
    _test()
