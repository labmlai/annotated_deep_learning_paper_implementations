"""
---
title: Relative Multi-Headed Attention
summary: >
  Documented implementation with explanations of
  Relative Multi-Headed Attention from paper Transformer-XL.
---

# Relative Multi-Headed Attention

This is an implementation of relative multi-headed attention from paper
[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
in [PyTorch](https://pytorch.org).
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

    # Reshape and remove excess elements from the end
    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
    x = x_padded[:-1].view_as(x)

    #
    return x


class RelativeMultiHeadAttention(MultiHeadAttention):
    """
    ## Relative Multi-Head Attention Module

    We override [Multi-Head Attention](mha.html) module so we only need to 
    write the `get_scores` method.
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        # The linear transformations do not need a bias since we
        # explicitly include it when calculating scores.
        # However having a bias for `value` might make sense.
        super().__init__(heads, d_model, dropout_prob, bias=False)

        # Number of relative positions
        self.P = 2 ** 12

        # Relative positional embeddings for key relative to the query.
        # We need $2P$ embeddings because the keys can be before or after the query.
        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P * 2, heads, self.d_k)), requires_grad=True)
        # Relative positional embedding bias for key relative to the query.
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P * 2, heads)), requires_grad=True)
        # Positional embeddings for the query is independent of the position of the query
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.d_k)), requires_grad=True)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        r"""
        ### Get relative attention scores

        With absolute attention

        \begin{align}
        A^{abs}_{j} &= lin_q(X^q_i + P_i)^\top lin_k(X^k_j + P_j) \\
                      &= \underset{\color{lightgreen}{A}}{Q_i^\top K_j} +
                         \underset{\color{lightgreen}{B}}{Q_i^\top U^K_j} +
                         \underset{\color{lightgreen}{C}}{{U^Q_i}^\top K_j} +
                         \underset{\color{lightgreen}{D}}{{U^Q_i}^\top U^K_j}
        \end{align}

        where $Q_i, K_j$, are linear transformations of
         original embeddings $X^q_i, X^k_j$
         and $U^Q_i, U^K_j$ are linear transformations of
         absolute positional encodings $P_i, P_j$.

        They reason out that the attention to a given key should be the same regardless of
        the position of query.
        Hence replace $\underset{\color{lightgreen}{C}}{{U^Q_i}^\top K_j}$
        with a constant $\underset{\color{lightgreen}{C}}{\color{orange}{v^\top} K_j}$.

        For the second and third terms relative positional encodings are introduced.
        So $\underset{\color{lightgreen}{B}}{Q_i^\top U^K_j}$ is
        replaced with $\underset{\color{lightgreen}{B}}{Q_i^\top \color{orange}{R_{i - j}}}$
        and $\underset{\color{lightgreen}{D}}{{U^Q_i}^\top U^K_j}$
        with $\underset{\color{lightgreen}{D}}{\color{orange}{S_{i-j}}}$.

        \begin{align}
        A^{rel}_{i,j} &= \underset{\mathbf{\color{lightgreen}{A}}}{Q_i^\top K_j} +
                         \underset{\mathbf{\color{lightgreen}{B}}}{Q_i^\top \color{orange}{R_{i - j}}} +
                         \underset{\mathbf{\color{lightgreen}{C}}}{\color{orange}{v^\top} K_j} +
                         \underset{\mathbf{\color{lightgreen}{D}}}{\color{orange}{S_{i-j}}}
        \end{align}
        """

        # $\color{orange}{R_k}$
        key_pos_emb = self.key_pos_embeddings[self.P - key.shape[0]:self.P + query.shape[0]]
        # $\color{orange}{S_k}$
        key_pos_bias = self.key_pos_bias[self.P - key.shape[0]:self.P + query.shape[0]]
        # $\color{orange}{v^\top}$
        query_pos_bias = self.query_pos_bias[None, None, :, :]

        # ${(\color{lightgreen}{\mathbf{A + C}})}_{i,j} =
        # Q_i^\top K_j +
        # \color{orange}{v^\top} K_jZ$
        ac = torch.einsum('ibhd,jbhd->ijbh', query + query_pos_bias, key)
        # $\color{lightgreen}{\mathbf{B'}_{i,k}} = Q_i^\top \color{orange}{R_k}$
        b = torch.einsum('ibhd,jhd->ijbh', query, key_pos_emb)
        # $\color{lightgreen}{\mathbf{D'}_{i,k}} = \color{orange}{S_k}$
        d = key_pos_bias[None, :, None, :]
        # Shift the rows of $\color{lightgreen}{\mathbf{(B' + D')}_{i,k}}$
        # to get $$\color{lightgreen}{\mathbf{(B + D)}_{i,j} = \mathbf{(B' + D')}_{i,i - j}}$$
        bd = shift_right(b + d)
        # Remove extra positions
        bd = bd[:, -key.shape[0]:]

        # Return the sum $$
        # \underset{\mathbf{\color{lightgreen}{A}}}{Q_i^\top K_j} +
        # \underset{\mathbf{\color{lightgreen}{B}}}{Q_i^\top \color{orange}{R_{i - j}}} +
        # \underset{\mathbf{\color{lightgreen}{C}}}{\color{orange}{v^\top} K_j} +
        # \underset{\mathbf{\color{lightgreen}{D}}}{\color{orange}{S_{i-j}}}
        # $$
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
