"""
---
title: Top-k Sampling
summary: A PyTorch implementation of top-k sampling from language models.
---

# Top-k Sampling

Here we first pick the top-k tokens from the distribution of logits, and then
sample from them.

Here's an [experiment](experiment.html) that uses these sampling techniques.
"""

import torch

from labml_nn.sampling import Sampler


class TopKSampler(Sampler):
    """
    ## Top-k Sampler
    """
    def __init__(self, k: int, sampler: Sampler):
        """
        :param k: is the number of tokens to pick
        :param sampler: is the sampler to use for the top-k tokens

        `sampler` can be any sampler that takes a logits tensor as input and returns a token tensor;
         e.g. [`TemperatureSampler'](temperature.html).
        """
        self.k = k
        self.sampler = sampler

    def __call__(self, logits: torch.Tensor):
        """
        Sample from logits
        """
        # New logits filled with $-\infty$; i.e. zero probability
        zeros = logits.new_ones(logits.shape) * float('-inf')
        # Pick the largest $k$ logits and their indices
        values, indices = torch.topk(logits, self.k, dim=-1)
        # Set the values of the top-k selected indices to actual logits.
        # Logits of other tokens remain $-\infty$
        zeros.scatter_(-1, indices, values)

        # Sample from the top-k logits with the specified sampler.
        return self.sampler(zeros)
