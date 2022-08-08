"""
---
title: Sampling Techniques for Language Models
summary: >
 A set of PyTorch implementations/tutorials of sampling techniques for language models.
---

# Sampling Techniques for Language Models

* [Greedy Sampling](greedy.html)
* [Temperature Sampling](temperature.html)
* [Top-k Sampling](top_k.html)
* [Nucleus Sampling](nucleus.html)

Here's an [experiment](experiment.html) that uses these sampling techniques.
"""

import torch


class Sampler:
    """
    ### Sampler base class
    """
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        ### Sample from logits

        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        """
        raise NotImplementedError()
