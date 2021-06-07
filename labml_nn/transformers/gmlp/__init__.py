from typing import Optional

import torch
from torch import nn


class SpacialGatingUnit(nn.Module):
    def __init__(self, d_channel, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm([d_channel])
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, u: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        seq_len = u.shape[0]

        if mask is not None:
            # `mask` has shape `[seq_len_q, seq_len_k, batch_size]`,
            # where first dimension is the query dimension.
            # If the query dimension is equal to $1$ it will be broadcasted.
            assert mask.shape[0] == 1 or mask.shape[0] == seq_len
            assert mask.shape[1] == seq_len
            # Same mask for all samples
            assert mask.shape[2] == 1
            mask = mask[:, :, 0]

        v = self.norm(v)
        weight, bias = self.proj.weight, self.proj.bias
        weight = weight[:seq_len, :seq_len]
        if mask is not None:
            weight = weight * mask

        v = torch.einsum('ij,jbd->ibd', weight, v) + bias[:seq_len, None, None]

        return u * v


class GMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm([d_model])
        self.activation = nn.GELU()
        self.proj1 = nn.Linear(d_model, d_ffn)
        self.sgu = SpacialGatingUnit(d_ffn // 2, seq_len)
        self.proj2 = nn.Linear(d_ffn // 2, d_model)
        self.size = d_model

    def forward(self, *, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        shortcut = x
        x = self.norm(x)
        x = self.proj1(x)
        x = self.activation(x)
        u, v = torch.chunk(x, 2, dim=-1)
        x = self.sgu(u, v, mask)
        x = self.proj2(x)

        return x + shortcut
