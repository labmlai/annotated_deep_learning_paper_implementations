"""
---
title: Weight Standardization
summary: >
 A PyTorch implementation/tutorial of Weight Standardization.
---

# Weight Standardization

This is a [PyTorch](https://pytorch.org) implementation of Weight Standardization from the paper
 [Micro-Batch Training with Batch-Channel Normalization and Weight Standardization](https://arxiv.org/abs/1903.10520).
We also have an [annotated implementation of Batch-Channel Normalization](../batch_channel_norm/index.html).

Batch normalization **gives a smooth loss landscape** and
**avoids elimination singularities**.
Elimination singularities are nodes of the network that become
useless (e.g. a ReLU that gives 0 all the time).

However, batch normalization doesn't work well when the batch size is too small,
which happens when training large networks because of device memory limitations.
The paper introduces Weight Standardization with Batch-Channel Normalization as
a better alternative.

Weight Standardization:
1. Normalizes the gradients
2. Smoothes the landscape (reduced Lipschitz constant)
3. Avoids elimination singularities

The Lipschitz constant is the maximum slope a function has between two points.
That is, $L$ is the Lipschitz constant where $L$ is the smallest value that satisfies,
$\forall a,b \in A: \lVert f(a) - f(b) \rVert \le L \lVert a - b \rVert$
where $f: A \rightarrow \mathbb{R}^m, A \in \mathbb{R}^n$.

Elimination singularities are avoided because it keeps the statistics of the outputs similar to the
inputs. So as long as the inputs are normally distributed the outputs remain close to normal.
This avoids outputs of nodes from always falling beyond the active range of the activation function
(e.g. always negative input for a ReLU).

*[Refer to the paper for proofs](https://arxiv.org/abs/1903.10520)*.

Here is [the training code](experiment.html) for training
a VGG network that uses weight standardization to classify CIFAR-10 data.
This uses a [2D-Convolution Layer with Weight Standardization](conv2d.html).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/normalization/weight_standardization/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/f4a783a2a7df11eb921d0242ac1c0002)
[![WandB](https://img.shields.io/badge/wandb-run-yellow)](https://wandb.ai/vpj/cifar10/runs/3flr4k8w)
"""

import torch


def weight_standardization(weight: torch.Tensor, eps: float):
    r"""
    ## Weight Standardization

    $$\hat{W}_{i,j} = \frac{W_{i,j} - \mu_{W_{i,\cdot}}} {\sigma_{W_{i,\cdot}}}$$

    where,

    \begin{align}
    W &\in \mathbb{R}^{O \times I} \\
    \mu_{W_{i,\cdot}} &= \frac{1}{I} \sum_{j=1}^I W_{i,j} \\
    \sigma_{W_{i,\cdot}} &= \sqrt{\frac{1}{I} \sum_{j=1}^I W^2_{i,j} - \mu^2_{W_{i,\cdot}} + \epsilon} \\
    \end{align}

    for a 2D-convolution layer $O$ is the number of output channels ($O = C_{out}$)
    and $I$ is the number of input channels times the kernel size ($I = C_{in} \times k_H \times k_W$)
    """

    # Get $C_{out}$, $C_{in}$ and kernel shape
    c_out, c_in, *kernel_shape = weight.shape
    # Reshape $W$ to $O \times I$
    weight = weight.view(c_out, -1)
    # Calculate
    #
    # \begin{align}
    # \mu_{W_{i,\cdot}} &= \frac{1}{I} \sum_{j=1}^I W_{i,j} \\
    # \sigma^2_{W_{i,\cdot}} &= \frac{1}{I} \sum_{j=1}^I W^2_{i,j} - \mu^2_{W_{i,\cdot}}
    # \end{align}
    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
    # Normalize
    # $$\hat{W}_{i,j} = \frac{W_{i,j} - \mu_{W_{i,\cdot}}} {\sigma_{W_{i,\cdot}}}$$
    weight = (weight - mean) / (torch.sqrt(var + eps))
    # Change back to original shape and return
    return weight.view(c_out, c_in, *kernel_shape)
