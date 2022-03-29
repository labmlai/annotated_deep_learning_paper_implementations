from typing import Union, List

import torch
from torch import nn, Size

from labml_nn.normalization.layer_norm import LayerNorm


class DeepNorm(nn.Module):
    def __init__(self, module: nn.Module, alpha: float, normalized_shape: Union[int, List[int], Size], *,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__()

        self.module = module
        self.alpha = alpha
        self.layer_norm = LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor):
        return x + self.alpha * self.module(x)


def deep_norm_init(weights: torch.Tensor, gain: float):
    # print(name)
    nn.init.xavier_normal_(weights, gain=gain)

