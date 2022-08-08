"""
---
title: Greedy Sampling
summary: A PyTorch implementation of greedy sampling from language models.
---

# Greedy Sampling

Here we sample the most likely token from the distribution of logits.

Here's an [experiment](experiment.html) that uses these sampling techniques.
"""

import torch

from labml_nn.sampling import Sampler


class GreedySampler(Sampler):
    def __call__(self, logits: torch.Tensor):
        """
        Sample the most likely token from the distribution of logits
        """
        return logits.argmax(dim=-1)
