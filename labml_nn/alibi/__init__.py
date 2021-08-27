"""
---
title: Attention with Linear Biases (ALiBi)
summary: >
  Documented implementation with explanations of Attention with Linear Biases (ALiBi)
---

# Attention with Linear Biases (ALiBi)

This is an implementation of Attention with Linear Biases (ALiBi).
"""
import math

import torch
from torch import nn

from labml.logger import inspect
from labml_nn.transformers.mha import MultiHeadAttention


def get_slopes(n_heads: int):
    """
    ## Get head-specific slope $m$ for each head
    """

    assert math.log2(n_heads).is_integer()

    s = (2 ** (-2 ** -(math.log2(n_heads) - 3)))
    r = s
    return [s * (r ** i) for i in range(n_heads)]


class AlibiMultiHeadAttention(MultiHeadAttention):
    """
    ## Attention with Linear Biases (ALiBi)

    We override [Multi-Head Attention](mha.html) module so we only need to
    write the `get_scores` method.
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        # The linear transformations do not need a bias since we
        # explicitly include it when calculating scores.
        # However having a bias for `value` might make sense.
        super().__init__(heads, d_model, dropout_prob)

        self.slopes = nn.Parameter(torch.tensor(get_slopes(heads)), requires_grad=False)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        r"""
        ### Calculate attention scores and add attention biases
        """

        # scores has shape `[query_seq_len, key_seq_len, batch_size, head]`
        scores = super().get_scores(query, key)

        distance = torch.arange(scores.shape[1]).to(scores.device, scores.dtype)
        bias = distance[None, :, None, None] * self.slopes[None, None, None, :]
        # add to scores
        scores = scores + bias

        return scores


def _test_slopes():
    inspect(get_slopes(8))
    inspect(get_slopes(16))


if __name__ == '__main__':
    _test_slopes()
