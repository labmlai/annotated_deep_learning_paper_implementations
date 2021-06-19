"""
---
title: "FNet: Mixing Tokens with Fourier Transforms"
summary: >
  This is an annotated implementation/tutorial of FNet in PyTorch.
---

# FNet: Mixing Tokens with Fourier Transforms

This is a [PyTorch](https://pytorch.org) implementation of the paper
[FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824).

This paper replaces the [self-attention layer](../mha.html) with two
[Fourier transforms](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) to
*mix* tokens.
This is a $7 \times$ more efficient than self-attention.
The accuracy loss of using this over self-attention is about 92% for
[BERT](https://paperswithcode.com/method/bert) on
[GLUE benchmark](https://paperswithcode.com/dataset/glue).

## Mixing tokens with two Fourier transforms

We apply Fourier transform along the hidden dimension (embedding dimension)
 and then along the sequence dimension.

$$
\mathcal{R}\big(\mathcal{F}_\text{seq} \big(\mathcal{F}_\text{hidden} (x) \big) \big)
$$

where $x$ is the embedding input, $\mathcal{F}$ stands for the fourier transform and
$\mathcal{R}$ stands for the real component in complex numbers.

This is very simple to implement on PyTorch - just 1 line of code.
The paper suggests using a precomputed DFT matrix and doing matrix multiplication to get the
Fourier transformation.

Here is [the training code](experiment.html) for using a FNet based model for classifying
[AG News](https://paperswithcode.com/dataset/ag-news).
"""

from typing import Optional

import torch
from torch import nn


class FNetMix(nn.Module):
    """
    ## FNet - Mix tokens

    This module simply implements
    $$
    \mathcal{R}\big(\mathcal{F}_\text{seq} \big(\mathcal{F}_\text{hidden} (x) \big) \big)
    $$

    The structure of this module is made similar to a [standard attention module](../mha.html) so that we can simply
    replace it.
    """

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        The [normal attention module](../mha.html) can be fed with different token embeddings for
        $\text{query}$,$\text{key}$, and $\text{value}$ and a mask.

        We follow the same function signature so that we can replace it directly.

        For FNet mixing, $$x = \text{query} = \text{key} = \text{value}$$ and masking is not possible.
        Shape of `query` (and `key` and `value`) is `[seq_len, batch_size, d_model]`.
        """

        # $\text{query}$,$\text{key}$, and $\text{value}$ all should be equal to $x$ for token mixing
        assert query is key and key is value
        # Token mixing doesn't support masking. i.e. all tokens will see all other token embeddings.
        assert mask is None

        # Assign to `x` for clarity
        x = query

        # Apply the Fourier transform along the hidden (embedding) dimension
        # $$\mathcal{F}_\text{hidden} (x)$$
        #
        # The output of the Fourier transform is a tensor of
        # [complex numbers](https://pytorch.org/docs/stable/complex_numbers.html).
        fft_hidden = torch.fft.fft(x, dim=2)
        # Apply the Fourier transform along the sequence dimension
        # $$\mathcal{F}_\text{seq} \big(\mathcal{F}_\text{hidden} (x) \big)$$
        fft_seq = torch.fft.fft(fft_hidden, dim=0)

        # Get the real component
        # $$\mathcal{R}\big(\mathcal{F}_\text{seq} \big(\mathcal{F}_\text{hidden} (x) \big) \big)$$
        return torch.real(fft_seq)
