"""
---
title: Patches Are All You Need? (ConvMixer)
summary: >
 A PyTorch implementation/tutorial of the paper
 "Patches Are All You Need?"
---

#  Patches Are All You Need? (ConvMixer)

"""

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.utils import clone_module_list


class PatchEmbeddings(Module):
    """
    <a id="PatchEmbeddings">
    ## Get patch embeddings
    </a>

    The paper splits the image into patches of equal size and do a linear transformation
    on the flattened pixels for each patch.

    We implement the same thing through a convolution layer, because it's simpler to implement.
    """

    def __init__(self, d_model: int, patch_size: int, in_channels: int):
        """
        * `d_model` is the transformer embeddings size
        * `patch_size` is the size of the patch
        * `in_channels` is the number of channels in the input image (3 for rgb)
        """
        super().__init__()

        # We create a convolution layer with a kernel size and and stride length equal to patch size.
        # This is equivalent to splitting the image into patches and doing a linear
        # transformation on each patch.
        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        # Apply convolution layer
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)

        # Return the patch embeddings
        return x


class ClassificationHead(Module):
    """
    <a id="ClassificationHead">
    ## MLP Classification Head
    </a>

    This is the two layer MLP head to classify the image based on `[CLS]` token embedding.
    """

    def __init__(self, d_model: int, n_classes: int):
        """
        * `d_model` is the transformer embedding size
        * `n_hidden` is the size of the hidden layer
        * `n_classes` is the number of classes in the classification task
        """
        super().__init__()
        # Average Pool
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Linear layer
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the transformer encoding for `[CLS]` token
        """
        # First layer and activation
        x = self.pool(x)
        x = x[:, :, 0, 0]
        # Linear layer
        x = self.linear(x)

        #
        return x


class ConvMixerLayer(Module):
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.depth_wise_conv = nn.Conv2d(d_model, d_model, kernel_size, groups=d_model, padding=(kernel_size - 1) // 2)
        self.act1 = nn.GELU()
        self.norm1 = nn.BatchNorm2d(d_model)

        self.point_wise_conv = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.act2 = nn.GELU()
        self.norm2 = nn.BatchNorm2d(d_model)

    def forward(self, x: torch.Tensor):
        residual = x

        x = self.depth_wise_conv(x)
        x = self.act1(x)
        x = self.norm1(x)

        x += residual

        x = self.point_wise_conv(x)
        x = self.act2(x)
        x = self.norm2(x)

        return x


class ConvMixer(Module):
    """
    ## ConvMixer
    """

    def __init__(self, conv_mixer_layer: ConvMixerLayer, n_layers: int,
                 patch_emb: PatchEmbeddings,
                 classification: ClassificationHead):
        """
        * `conv_mixer_layer` is a copy of a single ConvMixer layer.
         We make copies of it to make the transformer with `n_layers`.
        * `n_layers` is the number of [transformer layers](../models.html#TransformerLayer).
        * `patch_emb` is the [patch embeddings layer](#PatchEmbeddings).
        * `pos_emb` is the [positional embeddings layer](#LearnedPositionalEmbeddings).
        * `classification` is the [classification head](#ClassificationHead).
        """
        super().__init__()
        # Patch embeddings
        self.patch_emb = patch_emb
        # Classification head
        self.classification = classification
        # Make copies of the transformer layer
        self.conv_mixer_layer = clone_module_list(conv_mixer_layer, n_layers)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        # Get patch embeddings. This gives a tensor of shape `[patches, batch_size, d_model]`
        x = self.patch_emb(x)

        # Pass through transformer layers with no attention masking
        for layer in self.conv_mixer_layer:
            x = layer(x)

        # Classification head, to get logits
        x = self.classification(x)

        #
        return x
