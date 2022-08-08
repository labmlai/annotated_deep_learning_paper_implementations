"""
---
title: Sampling from Language Models with Temperature
summary: A PyTorch implementation of sampling from language models with temperature.
---

# Sampling from Language Models with Temperature

Here we sample from the following probability distribution where $V$ is the vocabulary,
$u_{1:|V|}$ are the logits of the distribution and T is the temperature:

$$P(x_i=V_l | x_{1:i-1}) = \frac{\exp(\frac{u_l}{T})}{\sum_j \exp(\frac{u_j}{T})}$$

$T = 1$ is normal random sampling.

Here's an [experiment](experiment.html) that uses these sampling techniques.
"""

import torch
from torch.distributions import Categorical

from labml_nn.sampling import Sampler


class TemperatureSampler(Sampler):
    """
    ## Sampler with Temperature
    """
    def __init__(self, temperature: float = 1.0):
        """
        :param temperature: is the temperature to sample with
        """
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor):
        """
        Sample from logits
        """

        # Create a categorical distribution with temperature adjusted logits
        dist = Categorical(logits=logits / self.temperature)

        # Sample
        return dist.sample()
