"""
---
title: U-Net model
summary: U-Net model
---

# U-Net model
"""
import torch
import torchvision.transforms.functional
from torch import nn


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.first(x)
        x = self.act(x)
        x = self.second(x)
        return self.act(x)


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConvolution(in_channels, out_channels)
        self.down = DownSample()

    def forward(self, x: torch.Tensor):
        x = self.down(x)
        x = self.conv(x)
        return x


class CopyAndCrop(nn.Module):
    def forward(self, x: torch.Tensor, pass_through: torch.Tensor):
        pass_through = torchvision.transforms.functional.center_crop(pass_through, [x.shape[2], x.shape[3]])
        # dh = pass_through.shape[2] - x.shape[2]
        # dw = pass_through.shape[3] - x.shape[3]
        #
        # pass_through = pass_through[:, :,
        #                dh // 2:dh // 2 + x.shape[2],
        #                dw // 2:dw // 2 + x.shape[3]
        #                ]

        x = torch.cat([x, pass_through], dim=1)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                        [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])

        self.middle_conv = DoubleConvolution(512, 1024)

        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                      [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.copy = nn.ModuleList([CopyAndCrop() for _ in range(4)])

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor):
        pass_through = []
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            pass_through.append(x)
            x = self.down_sample[i](x)

        x = self.middle_conv(x)

        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.copy[i](x, pass_through.pop())
            x = self.up_conv[i](x)

        x = self.final_conv(x)

        return x
