from typing import Optional, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from labml_nn.transformers.feed_forward import FeedForward


class MLPMixer(nn.Module):
    def __init__(self, ffn: 'FeedForward'):
        super().__init__()
        self.ffn = ffn

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        The [normal attention module](../mha.html) can be fed with different token embeddings for
        $\text{query}$,$\text{key}$, and $\text{value}$ and a mask.

        We follow the same function signature so that we can replace it directly.

        For MLP mixing, $$x = \text{query} = \text{key} = \text{value}$$ and masking is not possible.
        Shape of `query` (and `key` and `value`) is `[seq_len, batch_size, d_model]`.
        """

        # $\text{query}$,$\text{key}$, and $\text{value}$ all should be equal to $x$ for token mixing
        assert query is key and key is value
        # Token mixing doesn't support masking. i.e. all tokens will see all other token embeddings.
        assert mask is None

        # Assign to `x` for clarity
        x = query

        # Transpose
        x = x.transpose(0, 2)
        x = self.ffn(x)
        x = x.transpose(0, 2)

        return x
