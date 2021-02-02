"""
---
title: Fixed Positional Encodings
summary: >
  Implementation with explanation of fixed positional encodings as
  described in paper Attention is All You Need.
---

# Fixed Positional Encodings

The positional encoding encodes the position along the sequence into
 a vector of size `d_model`.

\begin{align}
PE_{p,2i} &= sin\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg) \\
PE_{p,2i + 1} &= cos\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)
\end{align}

Where $1 \leq 2i, 2i + 1 \leq d_{model}$
 are the feature indexes in the encoding, and $p$ is the position.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from labml_helpers.module import Module


class PositionalEncoding(Module):
    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len), False)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]].detach().requires_grad_(False)
        x = x + pe
        x = self.dropout(x)
        return x


def get_positional_encoding(d_model: int, max_len: int = 5000):
    # Empty encodings vectors
    encodings = torch.zeros(max_len, d_model)
    # Position indexes
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    # $2 * i$
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    # $10000^{\frac{2i}{d_{model}}$
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    # $PE_{p,2i} = sin\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 0::2] = torch.sin(position * div_term)
    # $PE_{p,2i + 1} = cos\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 1::2] = torch.cos(position * div_term)

    # Add batch dimension
    encodings = encodings.unsqueeze(1).requires_grad_(False)

    return encodings


def _test_positional_encoding():
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    pe = get_positional_encoding(20, 100)
    plt.plot(np.arange(100), pe[:, 0, 4:8].numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.title("Positional encoding")
    plt.show()


if __name__ == '__main__':
    _test_positional_encoding()
