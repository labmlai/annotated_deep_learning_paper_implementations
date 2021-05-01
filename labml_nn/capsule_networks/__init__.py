"""
---
title: Capsule Networks
summary: >
  PyTorch implementation and tutorial of Capsule Networks.
  Capsule network is a neural network architecture that embeds features
  as capsules and routes them with a voting mechanism to next layer of capsules.
---

# Capsule Networks

This is a [PyTorch](https://pytorch.org) implementation/tutorial of
[Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829).

Capsule network is a neural network architecture that embeds features
as capsules and routes them with a voting mechanism to next layer of capsules.

Unlike in other implementations of models, we've included a sample, because
it is difficult to understand some concepts with just the modules.
[This is the annotated code for a model that uses capsules to classify MNIST dataset](mnist.html)

This file holds the implementations of the core modules of Capsule Networks.

I used [jindongwang/Pytorch-CapsuleNet](https://github.com/jindongwang/Pytorch-CapsuleNet) to clarify some
confusions I had with the paper.

Here's a notebook for training a Capsule Network on MNIST dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/nn/blob/master/labml_nn/capsule_networks/mnist.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/e7c08e08586711ebb3e30242ac1c0002)
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from labml_helpers.module import Module


class Squash(Module):
    """
    ## Squash

    This is **squashing** function from paper, given by equation $(1)$.

    $$\mathbf{v}_j = \frac{{\lVert \mathbf{s}_j \rVert}^2}{1 + {\lVert \mathbf{s}_j \rVert}^2}
     \frac{\mathbf{s}_j}{\lVert \mathbf{s}_j \rVert}$$

    $\frac{\mathbf{s}_j}{\lVert \mathbf{s}_j \rVert}$
    normalizes the length of all the capsules, whilst
    $\frac{{\lVert \mathbf{s}_j \rVert}^2}{1 + {\lVert \mathbf{s}_j \rVert}^2}$
    shrinks the capsules that have a length smaller than one .
    """

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, s: torch.Tensor):
        """
        The shape of `s` is `[batch_size, n_capsules, n_features]`
        """

        # ${\lVert \mathbf{s}_j \rVert}^2$
        s2 = (s ** 2).sum(dim=-1, keepdims=True)

        # We add an epsilon when calculating $\lVert \mathbf{s}_j \rVert$ to make sure it doesn't become zero.
        # If this becomes zero it starts giving out `nan` values and training fails.
        # $$\mathbf{v}_j = \frac{{\lVert \mathbf{s}_j \rVert}^2}{1 + {\lVert \mathbf{s}_j \rVert}^2}
        # \frac{\mathbf{s}_j}{\sqrt{{\lVert \mathbf{s}_j \rVert}^2 + \epsilon}}$$
        return (s2 / (1 + s2)) * (s / torch.sqrt(s2 + self.epsilon))


class Router(Module):
    """
    ## Routing Algorithm

    This is the routing mechanism described in the paper.
    You can use multiple routing layers in your models.

    This combines calculating $\mathbf{s}_j$ for this layer and
    the routing algorithm described in *Procedure 1*.
    """

    def __init__(self, in_caps: int, out_caps: int, in_d: int, out_d: int, iterations: int):
        """
        `in_caps` is the number of capsules, and `in_d` is the number of features per capsule from the layer below.
        `out_caps` and `out_d` are the same for this layer.

        `iterations` is the number of routing iterations, symbolized by $r$ in the paper.
        """
        super().__init__()
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.iterations = iterations
        self.softmax = nn.Softmax(dim=1)
        self.squash = Squash()

        # This is the weight matrix $\mathbf{W}_{ij}$. It maps each capsule in the
        # lower layer to each capsule in this layer
        self.weight = nn.Parameter(torch.randn(in_caps, out_caps, in_d, out_d), requires_grad=True)

    def __call__(self, u: torch.Tensor):
        """
        The shape of `u` is `[batch_size, n_capsules, n_features]`.
        These are the capsules from the lower layer.
        """

        # $$\hat{\mathbf{u}}_{j|i} = \mathbf{W}_{ij} \mathbf{u}_i$$
        # Here $j$ is used to index capsules in this layer, whilst $i$ is
        # used to index capsules in the layer below (previous).
        u_hat = torch.einsum('ijnm,bin->bijm', self.weight, u)

        # Initial logits $b_{ij}$ are the log prior probabilities that capsule $i$
        # should be coupled with $j$.
        # We initialize these at zero
        b = u.new_zeros(u.shape[0], self.in_caps, self.out_caps)

        v = None

        # Iterate
        for i in range(self.iterations):
            # routing softmax $$c_{ij} = \frac{\exp({b_{ij}})}{\sum_k\exp({b_{ik}})}$$
            c = self.softmax(b)
            # $$\mathbf{s}_j = \sum_i{c_{ij} \hat{\mathbf{u}}_{j|i}}$$
            s = torch.einsum('bij,bijm->bjm', c, u_hat)
            # $$\mathbf{v}_j = squash(\mathbf{s}_j)$$
            v = self.squash(s)
            # $$a_{ij} = \mathbf{v}_j \cdot \hat{\mathbf{u}}_{j|i}$$
            a = torch.einsum('bjm,bijm->bij', v, u_hat)
            # $$b_{ij} \gets b_{ij} + \mathbf{v}_j \cdot \hat{\mathbf{u}}_{j|i}$$
            b = b + a

        return v


class MarginLoss(Module):
    """
    ## Margin loss for class existence

    A separate margin loss is used for each output capsule and the total loss is the sum of them.
    The length of each output capsule is the probability that class is present in the input.

    Loss for each output capsule or class $k$ is,
    $$\mathcal{L}_k = T_k \max(0, m^{+} - \lVert\mathbf{v}_k\rVert)^2 +
    \lambda (1 - T_k) \max(0, \lVert\mathbf{v}_k\rVert - m^{-})^2$$

    $T_k$ is $1$ if the class $k$ is present and $0$ otherwise.
    The first component of the loss is $0$ when the class is not present,
    and the second component is $0$ if the class is present.
    The $\max(0, x)$ is used to avoid predictions going to extremes.
    $m^{+}$ is set to be $0.9$ and $m^{-}$ to be $0.1$ in the paper.

    The $\lambda$ down-weighting is used to stop the length of all capsules from
    falling during the initial phase of training.
    """
    def __init__(self, *, n_labels: int, lambda_: float = 0.5, m_positive: float = 0.9, m_negative: float = 0.1):
        super().__init__()

        self.m_negative = m_negative
        self.m_positive = m_positive
        self.lambda_ = lambda_
        self.n_labels = n_labels

    def __call__(self, v: torch.Tensor, labels: torch.Tensor):
        """
        `v`, $\mathbf{v}_j$ are the squashed output capsules.
        This has shape `[batch_size, n_labels, n_features]`; that is, there is a capsule for each label.

        `labels` are the labels, and has shape `[batch_size]`.
        """
        # $$\lVert \mathbf{v}_j \rVert$$
        v_norm = torch.sqrt((v ** 2).sum(dim=-1))

        # $$\mathcal{L}$$
        # `labels` is one-hot encoded labels of shape `[batch_size, n_labels]`
        labels = torch.eye(self.n_labels, device=labels.device)[labels]

        # $$\mathcal{L}_k = T_k \max(0, m^{+} - \lVert\mathbf{v}_k\rVert)^2 +
        # \lambda (1 - T_k) \max(0, \lVert\mathbf{v}_k\rVert - m^{-})^2$$
        # `loss` has shape `[batch_size, n_labels]`. We have parallelized the computation
        # of $\mathcal{L}_k$ for for all $k$.
        loss = labels * F.relu(self.m_positive - v_norm) + \
               self.lambda_ * (1.0 - labels) * F.relu(v_norm - self.m_negative)

        # $$\sum_k \mathcal{L}_k$$
        return loss.sum(dim=-1).mean()
