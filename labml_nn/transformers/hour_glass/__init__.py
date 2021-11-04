from typing import List

import torch
from labml_nn.transformers.utils import subsequent_mask
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers import MultiHeadAttention, TransformerLayer
from labml_nn.transformers.feed_forward import FeedForward


class ShiftRight(Module):
    def __init__(self, shift: int):
        super().__init__()
        assert shift >= 0
        self.shift = shift

    def forward(self, x: torch.Tensor):
        if self.shift == 0:
            return x
        prefix = x.new_zeros([self.shift, *x.shape[1:]])
        return torch.cat([prefix, x[:-self.shift]])


class AvgPoolShortening(Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor):
        rem = len(x) % self.k
        if rem != 0:
            x_rem = x[-rem:]
            if rem < len(x):
                x = x[:-rem]
            else:
                x = None
        else:
            x_rem = None

        if x is not None:
            x = x.reshape(x.shape[0] // self.k, self.k, *x.shape[1:])
            x = x.mean(dim=1)

        if x_rem is not None:
            x_rem = x_rem.mean(dim=0, keepdim=True)

        if x is None:
            return x_rem

        if x_rem is None:
            return x

        return torch.cat([x, x_rem])


class NaiveUpSampling(Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor, x_short: torch.Tensor):
        expanded = torch.repeat_interleave(x_short, self.k, dim=0)
        expanded = expanded[:x.shape[0]]

        return expanded


class AutoregressiveMask(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x: torch.Tensor):
        if self.mask is None or self.mask.size(0) != len(x):
            # Subsequent mask, will mask out tokens from seeing future tokens
            self.mask = subsequent_mask(len(x)).to(x.device)

        return self.mask


class HourGlass(Module):
    def __init__(self, n_heads: int, d_model: int, dropout: float, d_ff: int, shortening_factors: List[int]):
        super().__init__()
        self.pre = TransformerLayer(d_model=d_model,
                                    self_attn=MultiHeadAttention(n_heads, d_model, dropout),
                                    feed_forward=FeedForward(d_model, d_ff, dropout),
                                    dropout_prob=dropout)

        k = shortening_factors[0]

        self.shift_right = ShiftRight(k - 1)
        self.shortening = AvgPoolShortening(k)

        self.pre = TransformerLayer(d_model=d_model,
                                    self_attn=MultiHeadAttention(n_heads, d_model, dropout),
                                    feed_forward=FeedForward(d_model, d_ff, dropout),
                                    dropout_prob=dropout)

        if len(shortening_factors) == 1:
            self.shortened = TransformerLayer(d_model=d_model,
                                              self_attn=MultiHeadAttention(n_heads, d_model, dropout),
                                              feed_forward=FeedForward(d_model, d_ff, dropout),
                                              dropout_prob=dropout)
            self.mask_short = AutoregressiveMask()
            self.hour_glass = None
        else:
            self.hour_glass = HourGlass(n_heads, d_model, dropout, d_ff, shortening_factors[1:])

        self.up_sampling = NaiveUpSampling(k)
        self.post = TransformerLayer(d_model=d_model,
                                     self_attn=MultiHeadAttention(n_heads, d_model, dropout),
                                     feed_forward=FeedForward(d_model, d_ff, dropout),
                                     dropout_prob=dropout)

        self.mask = AutoregressiveMask()

    def forward(self, x: torch.Tensor):
        x = self.pre(x=x, mask=self.mask(x))
        x_short = self.shortening(self.shift_right(x))
        if self.hour_glass is None:
            x_short = self.shortened(x=x_short, mask=self.mask_short(x_short))
        else:
            x_short = self.hour_glass(x_short)

        x = x + self.up_sampling(x, x_short)
        x = self.post(x=x, mask=self.mask(x))

        return x
