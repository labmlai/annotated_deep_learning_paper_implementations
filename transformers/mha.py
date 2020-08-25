import math
from typing import Optional

import torch
from torch import nn as nn
from torch.nn import functional as F

from labml.helpers.pytorch.module import Module


class PrepareForMultiHeadAttention(Module):
    def __init__(self, d_model: int, heads: int, d_k: int):
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k)
        self.heads = heads
        self.d_k = d_k

    def __call__(self, x: torch.Tensor):
        seq_len, batch_size, _ = x.shape

        x = self.linear(x)
        x = x.view(seq_len, batch_size, self.heads, self.d_k)

        return x


class MultiHeadAttention(Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__()
        # We assume d_v always equals d_k
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k)
        self.output = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)

    def get_scores(self, query: torch.Tensor,
                   key: torch.Tensor, ):
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def __call__(self, *,
                 query: torch.Tensor,
                 key: torch.Tensor,
                 value: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        seq_len, batch_size, *_ = query.shape

        if mask is not None:
            # mask = ijb
            assert mask.shape[0] == 1 or mask.shape[0] == mask.shape[1]
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(-1)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query, key)

        scores *= self.scale
        if mask is not None:
            # mask = ijbh
            assert mask.shape[0] == 1 or mask.shape[0] == mask.shape[1]
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=1)
        attn = self.dropout(attn)

        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        self.attn = attn.detach()

        x = x.reshape(seq_len, batch_size, -1)

        return self.output(x)
