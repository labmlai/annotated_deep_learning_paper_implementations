#!/bin/python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary

#custom import
import numpy as np
import time
import os


# ResBlock
class ResBlock(nn.Module):
    def __init__(self, num_features, use_batch_norm=False):
        super(ResBlock, self).__init__()
        self.num_features = num_features
        self.conv_layer1 = nn.Conv2d(num_features, num_features,  kernel_size=3, stride=1, padding=1)
        self.relu_layer = nn.ReLU()
        self.conv_layer2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)

        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.batch_norm_layer1 = nn.BatchNorm2d(self.num_features)
            self.batch_norm_layer2 = nn.BatchNorm2d(self.num_features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        residual = x
        x = self.conv_layer1(x)
        if self.use_batch_norm:
            x = self.batch_norm_layer1(x)

        x = self.relu_layer(x)
        x = self.conv_layer2(x)
        if self.use_batch_norm:
            x = self.batch_norm_layer2(x)

        x += residual
        x = self.relu_layer(x)
        return x

# ResNet
class ResNet(nn.Module):
    def __init__(self, in_features, num_class, feature_channel_list, batch_norm= False, num_stacks=1, zero_init_residual=True):
        super(ResNet, self).__init__()
        self.in_features = in_features
        self.num_in_channel = in_features[2]
        self.num_class = num_class
        self.feature_channel_list = feature_channel_list
        self.num_residual_blocks = len(self.feature_channel_list)
        self.num_stacks = num_stacks
        self.batch_norm = batch_norm
        self.shape_list = []
        self.shape_list.append(in_features)
        self.module_list = nn.ModuleList()
        self.zero_init_residual= zero_init_residual
        self.build_()

    def build_(self):
        #track filter shape
        cur_shape = self.GetCurShape()
        cur_shape = self.CalcConvOutShape(cur_shape, kernel_size=7, padding=1, stride=2, out_filters= self.feature_channel_list[0])
        self.shape_list.append(cur_shape)

        if len(self.in_features) == 2:
            in_channels = 1
        else:
            in_channels = self.in_features[2]

        # First Conv layer 7x7 stride=2, pad =1
        self.module_list.append(nn.Conv2d(in_channels= in_channels,
                                    out_channels= self.feature_channel_list[0],
                                    kernel_size=7,
                                    stride=2,
                                    padding=3))


        #batch norm
        if self.batch_norm: #batch_norm
            self.module_list.append(nn.BatchNorm2d(self.feature_channel_list[0]))

        # ReLU()
        self.module_list.append(nn.ReLU())

        for i in range(self.num_residual_blocks-1):
            in_size = self.feature_channel_list[i]
            out_size = self.feature_channel_list[i+1]

            res_block = ResBlock(in_size, use_batch_norm=True)

            # #Stacking Residual blocks
            for num in range(self.num_stacks):
                self.module_list.append(res_block)

            # # Intermediate Conv and ReLU()
            self.module_list.append(nn.Conv2d(in_channels=in_size,
                                              out_channels= out_size,
                                              kernel_size=3,
                                              padding=1,
                                              stride=2))

            # track filter shape
            cur_shape = self.CalcConvOutShape(cur_shape, kernel_size=3, padding=1,
                                         stride=2, out_filters=out_size)

            self.shape_list.append(cur_shape)

            # # batch norm
            if self.batch_norm:  # batch_norm
                self.module_list.append(nn.BatchNorm2d(out_size))

            self.module_list.append(nn.ReLU())

            # print("shape list", self.shape_list)

        #TODO include in the main loop
        #Last Residual block
        res_block = ResBlock(out_size, use_batch_norm=True)
        for num in range(self.num_stacks):
            self.module_list.append(res_block)

        #Last AvgPool layer
        # self.module_list.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        self.module_list.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # track filter shape
        cur_shape = self.CalcConvOutShape(cur_shape, kernel_size=2, padding=0, stride=2, out_filters=out_size)
        self.shape_list.append(cur_shape)

        s = self.GetCurShape()
        in_features = s[0] * s[1] * s[2]

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.xavier_uniform_(m.weight)

        # if self.zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, ResBlock):
        #             nn.init.constant_(m.batch_norm_layer1.weight, 0)
        #             nn.init.constant_(m.batch_norm_layer2.weight, 0)

    def GetCurShape(self):
        return self.shape_list[-1]

    def CalcConvFormula(self, W, K, P, S):
        return int(np.floor(((W - K + 2 * P) / S) + 1))

    # https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer
    # Calculate the output shape after applying a convolution
    def CalcConvOutShape(self, in_shape, kernel_size, padding, stride, out_filters):
        # Multiple options for different kernel shapes
        if type(kernel_size) == int:
            out_shape = [self.CalcConvFormula(in_shape[i], kernel_size, padding, stride) for i in range(2)]
        else:
            out_shape = [self.CalcConvFormula(in_shape[i], kernel_size[i], padding, stride) for i in range(2)]

        return (out_shape[0], out_shape[1], out_filters)  # , batch_size... but not necessary.

    def AddMLP(self, MLP):
        if MLP:
            self.module_list.append(MLP)

    # def MLP(self, in_features, num_classes, use_batch_norm=False, use_dropout=False, use_softmax=False):
    #     return nn.ReLU(nn.Linear(in_features, num_classes))

    def forward(self, x):
        for mod_name in self.module_list:
            x = mod_name(x)
        x = x.view(x.size(0), -1)  # flat #TODO check if it works
        return x


