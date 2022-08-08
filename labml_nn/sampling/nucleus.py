"""
---
title: Nucleus Sampling
summary: A PyTorch implementation of nucleus sampling from language models.
---

# Nucleus Sampling

This is an implementation of nucleus sampling, introduced in the paper
[The Curious Case of Neural Text Degeneration](https://papers.labml.ai/paper/1904.09751).

The paper discusses the problems with other sampling methods such as Beam Search,
[Pure sampling](temperature.html), [Temperature sampling](temperature.html), and
[Top-k sampling](top_k.html). The paper introduces the idea of nucleus sampling,
which practically performs better than other sampling methods for text generation.

Nucleus sampling first picks a subset of the vocabulary $V^{(p)} \subset V$,
where $V^{(p)}$ is smallest set of tokens such that

$$\sum_{x_i \in V^{(p)}} P(x_i | x_{1:i-1}) \ge p$$

That is, we pick the highest probable tokens until the sum of their probabilities is less that $p$.

Then we sample from the selected tokens.

Here's an [experiment](experiment.html) that uses these sampling techniques.
"""

import torch
from torch import nn

from labml_nn.sampling import Sampler


class NucleusSampler(Sampler):
    """
    ## Nucleus Sampler
    """
    def __init__(self, p: float, sampler: Sampler):
        """
        :param p: is the sum of probabilities of tokens to pick $p$
        :param sampler: is the sampler to use for the selected tokens
        """
        self.p = p
        self.sampler = sampler
        # Softmax to compute $P(x_i | x_{1:i-1})$ from the logits
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, logits: torch.Tensor):
        """
        Sample from logits with Nucleus Sampling
        """

        # Get probabilities $P(x_i | x_{1:i-1})$
        probs = self.softmax(logits)

        # Sort probabilities in descending order
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        # Get the cumulative sum of probabilities in the sorted order
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
        # Find the cumulative sums less than $p$.
        nucleus = cum_sum_probs < self.p
        # Prepend ones so that we add one token after the minimum number
        # of tokens with cumulative probability less that $p$.
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)

        # Get log probabilities and mask out the non-nucleus
        sorted_log_probs = torch.log(sorted_probs)
        sorted_log_probs[~nucleus] = float('-inf')

        # Sample from the sampler
        sampled_sorted_indexes = self.sampler(sorted_log_probs)

        # Get the actual indexes
        res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))

        #
        return res.squeeze(-1)
