"""
# Relative Multi-head Attention

This is an implementation of 
[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
"""

import torch
from torch import nn

from labml.logger import inspect
from labml_nn.transformers.mha import MultiHeadAttention


def shift_right(x: torch.Tensor):
    """
    This method shifts $i^{th}$ row of a matrix by $i$ columns.

    If the input is `[[1, 2 ,3], [4, 5 ,6], [7, 8, 9]]`, the shifted
    result would be `[[1, 2 ,3], [0, 4, 5], [9, 0, 7]]`.
    *Ideally we should mask out the lower triangle but it's ok for our purpose*.
    """

    # Concatenate a column of zeros
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    x_padded = torch.cat([x, zero_pad], dim=1)

    # Remove excess elements from the end
    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
    x = x_padded[:-1].view_as(x)

    return x


class RelativeMultiHeadAttention(MultiHeadAttention):
    """
    ## Relative Multi-Head Attention Module

    We override [Multi-Head Attention](mha.html) module so we only need to 
    write the `get_scores` method.
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        # The linear transformations doesn't need a bias since we take care of it when
        # calculating scores.
        # However having a bias for `value` might make sense.
        super().__init__(heads, d_model, dropout_prob, False)

        self.P = 2 ** 12

        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P * 2, heads, self.d_k)), requires_grad=True)
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.d_k)), requires_grad=True)
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P * 2, heads)), requires_grad=True)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        With absolute attention

        \begin{align}
        A^{abs}_{i,j} &= lin_q(X^q_i + P_i)^T lin_k(X^k_j + P_j) \\
                      &= Q_i^T K_j + Q_i^T U_j + V_i^T K_j + V_i^T U_j
        \end{align}

        where $Q_i$, $K_j$, $V_i$, and $U_j$ are linear transformations of
         orginal embeddings and positional encodings.

        They reason out that the attention to a given key should be the same regardless of 
        the position of query. Hence replace $V_i^T K_j$ with a constant $v^T K_j$.
        🤔 May be worthwhile testing without this assumption.

        For the second and third terms relative positional encodings are introduced.
        So $Q_i^T U_j$ is replaced with $Q_i^T R_{i - j}$ and $V_i^T U_j$ with $S_{i-j}$.

        \begin{align}
        A^{rel}_{i,j} &= Q_i^T K_j + Q_i^T R_{i - j} + v^T K_j + S_{i-j}
        \end{align}

        """

        # $R_{i-j}$ pre-shift
        key_pos_emb = self.key_pos_embeddings[self.P - query.shape[0]:self.P + key.shape[0]]
        # $S_{i-j}$ pre-shift
        key_pos_bias = self.key_pos_bias[self.P - query.shape[0]:self.P + key.shape[0]]
        # $v^T$
        query_pos_bias = self.query_pos_bias[None, None, :, :]

        # $Q_i^T K_j + v^T K_j$
        ac = torch.einsum('ibhd,jbhd->ijbh', query + query_pos_bias, key)
        # $Q_i^T R_{i - j}$ pre-shift
        b = torch.einsum('ibhd,jhd->ijbh', query, key_pos_emb)
        # $S_{i-j}$ pre-shift
        d = key_pos_bias[None, :, None, :]
        # $Q_i^T R_{i - j} + S_{i-j}$
        bd = shift_right(b + d)
        bd = bd[:, -key.shape[0]:]

        return ac + bd


def _test_shift_right():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    inspect(x)
    inspect(shift_right(x))

    x = torch.arange(1, 6)[None, :, None, None].repeat(5, 1, 1, 1)
    inspect(x[:, :, 0, 0])
    inspect(shift_right(x)[:, :, 0, 0])

    x = torch.arange(1, 6)[None, :, None, None].repeat(3, 1, 1, 1)
    inspect(x[:, :, 0, 0])
    inspect(shift_right(x)[:, :, 0, 0])


if __name__ == '__main__':
    _test_shift_right()
