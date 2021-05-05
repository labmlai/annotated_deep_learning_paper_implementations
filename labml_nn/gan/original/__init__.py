"""
---
title: Generative Adversarial Networks (GAN)
summary: A simple PyTorch implementation/tutorial of Generative Adversarial Networks (GAN) loss functions.
---

# Generative Adversarial Networks (GAN)

This is an implementation of
[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661).

The generator, $G(\pmb{z}; \theta_g)$ generates samples that match the
distribution of data, while the discriminator, $D(\pmb{x}; \theta_g)$
gives the probability that $\pmb{x}$ came from data rather than $G$.

We train $D$ and $G$ simultaneously on a two-player min-max game with value
function $V(G, D)$.

$$\min_G \max_D V(D, G) =
    \mathop{\mathbb{E}}_{\pmb{x} \sim p_{data}(\pmb{x})}
        \big[\log D(\pmb{x})\big] +
    \mathop{\mathbb{E}}_{\pmb{z} \sim p_{\pmb{z}}(\pmb{z})}
        \big[\log (1 - D(G(\pmb{z}))\big]
$$

$p_{data}(\pmb{x})$ is the probability distribution over data,
whilst $p_{\pmb{z}}(\pmb{z})$ probability distribution of $\pmb{z}$, which is set to
gaussian noise.

This file defines the loss functions. [Here](../simple_mnist_experiment.html) is an MNIST example
with two multilayer perceptron for the generator and discriminator.
"""

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data

from labml_helpers.module import Module


class DiscriminatorLogitsLoss(Module):
    """
    ## Discriminator Loss

    Discriminator should **ascend** on the gradient,

    $$\nabla_{\theta_d} \frac{1}{m} \sum_{i=1}^m \Bigg[
        \log D\Big(\pmb{x}^{(i)}\Big) +
        \log \Big(1 - D\Big(G\Big(\pmb{z}^{(i)}\Big)\Big)\Big)
    \Bigg]$$

    $m$ is the mini-batch size and $(i)$ is used to index samples in the mini-batch.
    $\pmb{x}$ are samples from $p_{data}$ and $\pmb{z}$ are samples from $p_z$.
    """

    def __init__(self, smoothing: float = 0.2):
        super().__init__()
        # We use PyTorch Binary Cross Entropy Loss, which is
        # $-\sum\Big[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})\Big]$,
        # where $y$ are the labels and $\hat{y}$ are the predictions.
        # *Note the negative sign*.
        # We use labels equal to $1$ for $\pmb{x}$ from $p_{data}$
        # and labels equal to $0$ for $\pmb{x}$ from $p_{G}.$
        # Then descending on the sum of these is the same as ascending on
        # the above gradient.
        #
        # `BCEWithLogitsLoss` combines softmax and binary cross entropy loss.
        self.loss_true = nn.BCEWithLogitsLoss()
        self.loss_false = nn.BCEWithLogitsLoss()

        # We use label smoothing because it seems to work better in some cases
        self.smoothing = smoothing

        # Labels are registered as buffered and persistence is set to `False`.
        self.register_buffer('labels_true', _create_labels(256, 1.0 - smoothing, 1.0), False)
        self.register_buffer('labels_false', _create_labels(256, 0.0, smoothing), False)

    def __call__(self, logits_true: torch.Tensor, logits_false: torch.Tensor):
        """
        `logits_true` are logits from $D(\pmb{x}^{(i)})$ and
        `logits_false` are logits from $D(G(\pmb{z}^{(i)}))$
        """
        if len(logits_true) > len(self.labels_true):
            self.register_buffer("labels_true",
                                 _create_labels(len(logits_true), 1.0 - self.smoothing, 1.0, logits_true.device), False)
        if len(logits_false) > len(self.labels_false):
            self.register_buffer("labels_false",
                                 _create_labels(len(logits_false), 0.0, self.smoothing, logits_false.device), False)

        return (self.loss_true(logits_true, self.labels_true[:len(logits_true)]),
                self.loss_false(logits_false, self.labels_false[:len(logits_false)]))


class GeneratorLogitsLoss(Module):
    """
    ## Generator Loss

    Generator should **descend** on the gradient,

    $$\nabla_{\theta_g} \frac{1}{m} \sum_{i=1}^m \Bigg[
        \log \Big(1 - D\Big(G\Big(\pmb{z}^{(i)}\Big)\Big)\Big)
    \Bigg]$$
    """
    def __init__(self, smoothing: float = 0.2):
        super().__init__()
        self.loss_true = nn.BCEWithLogitsLoss()
        self.smoothing = smoothing
        # We use labels equal to $1$ for $\pmb{x}$ from $p_{G}.$
        # Then descending on this loss is the same as descending on
        # the above gradient.
        self.register_buffer('fake_labels', _create_labels(256, 1.0 - smoothing, 1.0), False)

    def __call__(self, logits: torch.Tensor):
        if len(logits) > len(self.fake_labels):
            self.register_buffer("fake_labels",
                                 _create_labels(len(logits), 1.0 - self.smoothing, 1.0, logits.device), False)

        return self.loss_true(logits, self.fake_labels[:len(logits)])


def _create_labels(n: int, r1: float, r2: float, device: torch.device = None):
    """
    Create smoothed labels
    """
    return torch.empty(n, 1, requires_grad=False, device=device).uniform_(r1, r2)
