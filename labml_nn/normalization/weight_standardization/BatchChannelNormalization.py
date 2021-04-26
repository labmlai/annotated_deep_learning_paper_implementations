import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.normalization.batch_norm import BatchNorm


class ChannelNorm(Module):
    """
    This is similar to group norm but affine transform is done group wise
    """

    def __init__(self, num_channels, num_groups,
                 eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine
        # Note that these transforms are per group, unlike in group norm where
        # they are transformed channel-wise
        if self.affine:
            self.scale = nn.Parameter(torch.ones(num_groups))
            self.shift = nn.Parameter(torch.zeros(num_groups))

    def __call__(self, x: torch.Tensor):
        # Keep the original shape
        x_shape = x.shape
        # Get the batch size
        batch_size = x_shape[0]
        # Sanity check to make sure the number of features is the same
        assert self.channels == x.shape[1]

        # Reshape into `[batch_size, groups, n]`
        x = x.view(batch_size, self.groups, -1)

        # Calculate the mean across last dimension;
        # i.e. the means for each sample and channel group $\mathbb{E}[x_{(i_N, i_G)}]$
        mean = x.mean(dim=[-1], keepdim=True)
        # Calculate the squared mean across last dimension;
        # i.e. the means for each sample and channel group $\mathbb{E}[x^2_{(i_N, i_G)}]$
        mean_x2 = (x ** 2).mean(dim=[-1], keepdim=True)
        # Variance for each sample and feature group
        # $Var[x_{(i_N, i_G)}] = \mathbb{E}[x^2_{(i_N, i_G)}] - \mathbb{E}[x_{(i_N, i_G)}]^2$
        var = mean_x2 - mean ** 2

        # Normalize
        # $$\hat{x}_{(i_N, i_G)} =
        # \frac{x_{(i_N, i_G)} - \mathbb{E}[x_{(i_N, i_G)}]}{\sqrt{Var[x_{(i_N, i_G)}] + \epsilon}}$$
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift channel-wise
        # $$y_{i_C} =\gamma_{i_C} \hat{x}_{i_C} + \beta_{i_C}$$
        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        # Reshape to original and return
        return x_norm.view(x_shape)


class BatchChannelNorm(Module):
    def __init__(self, num_channels, num_groups, eps: float = 1e-5,
                 momentum: float = 0.1, estimate: bool = False):
        super().__init__()
        if estimate:
            self.batch_norm = EstimatedBatchNorm(num_channels,
                                                 eps=eps, momentum=momentum)
        else:
            self.batch_norm = BatchNorm(num_channels,
                                        eps=eps, momentum=momentum)

        self.channel_norm = ChannelNorm(num_channels, num_groups, eps)

    def __call__(self, x):
        x = self.batch_norm(x)
        return self.channel_norm(x)


class EstimatedBatchNorm(Module):
    def __init__(self, num_features, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.num_features = num_features
        if self.affine:
            self.scale = nn.Parameter(torch.ones(num_features))
            self.shift = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('exp_mean', torch.zeros(num_features))
        self.register_buffer('exp_var', torch.ones(num_features))

    def __call__(self, x):
        x_shape = x.shape
        # Get the batch size
        batch_size = x_shape[0]
        # Reshape into `[batch_size, channels, n]`
        x = x.view(batch_size, self.channels, -1)

        if self.training:
            with torch.no_grad():
                # Calculate the mean across first and last dimension;
                # i.e. the means for each feature $\mathbb{E}[x^{(k)}]$
                mean = x.mean(dim=[0, 2])
                # Calculate the squared mean across first and last dimension;
                # i.e. the means for each feature $\mathbb{E}[(x^{(k)})^2]$
                mean_x2 = (x ** 2).mean(dim=[0, 2])
                # Variance for each feature $Var[x^{(k)}] = \mathbb{E}[(x^{(k)})^2] - \mathbb{E}[x^{(k)}]^2$
                var = mean_x2 - mean ** 2

                # Update exponential moving averages
                self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
                self.exp_var = (1 - self.momentum) * self.exp_var + self.momentum * var

        # Normalize $$\hat{x}^{(k)} = \frac{x^{(k)} - \mathbb{E}[x^{(k)}]}{\sqrt{Var[x^{(k)}] + \epsilon}}$$
        x_norm = (x - self.exp_mean.view(1, -1, 1)) / torch.sqrt(self.exp_var + self.eps).view(1, -1, 1)
        # Scale and shift $$y^{(k)} =\gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$$
        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        # Reshape to original and return
        return x_norm.view(x_shape)
