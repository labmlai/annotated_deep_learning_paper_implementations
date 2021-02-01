import torch
from torch import nn

from labml_helpers.module import Module


class BatchNorm(Module):
    def __init__(self, channels: int, *,
                 eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        super().__init__()

        self.channels = channels

        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(channels))
            self.register_buffer('running_var', torch.ones(channels))

    def __call__(self, x: torch.Tensor):
        x_shape = x.shape
        batch_size = x_shape[0]

        x = x.view(batch_size, self.channels, -1)
        if self.training or not self.track_running_stats:
            mean = x.mean(dim=[0, 2])
            mean_x2 = (x ** 2).mean(dim=[0, 2])
            var = mean_x2 - mean ** 2

            if self.training and self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean.view(1, -1, 1)) / torch.sqrt(var + self.eps).view(1, -1, 1)
        if self.affine:
            x_norm = self.weight.view(1, -1, 1) * x_norm + self.bias.view(1, -1, 1)

        return x_norm.view(x_shape)
