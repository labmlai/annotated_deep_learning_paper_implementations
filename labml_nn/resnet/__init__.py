from typing import List

import torch
from torch import nn

from labml_helpers.module import Module


class ShortcutProjection(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def __call__(self, x: torch.Tensor):
        return self.bn(self.conv(x))


class ResidualBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

    def __call__(self, x: torch.Tensor):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act2(x + shortcut)


class ResNetBase(Module):
    def __init__(self, n_blocks: List[int], n_channels: List[int], img_channels: int = 3, first_kernel_size: int = 7):
        super().__init__()

        assert len(n_blocks) == len(n_channels)
        self.conv = nn.Conv2d(img_channels, n_channels[0],
                              kernel_size=first_kernel_size, stride=1, padding=first_kernel_size // 2)
        self.bn = nn.BatchNorm2d(n_channels[0])

        blocks = []
        prev_channels = n_channels[0]
        for n_b, channels in zip(n_blocks, n_channels):
            blocks.append(ResidualBlock(prev_channels, channels, stride=2 if len(blocks) == 0 else 1))
            prev_channels = channels
            for i in range(n_b - 1):
                blocks.append(ResidualBlock(channels, channels, stride=1))

        self.blocks = nn.Sequential(*blocks)

    def __call__(self, x: torch.Tensor):
        """
        x has shape [batch_size, img_channels, height, width]
        """

        x = self.bn(self.conv(x))
        x = self.blocks(x)
        # x to shape `[batch_size, channels, H * W]`
        x = x.view(x.shape[0], x.shape[1], -1)
        # Global average pooling
        return x.mean(dim=-1)
