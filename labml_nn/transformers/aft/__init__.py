from typing import Optional

import torch
from torch import nn

from labml_helpers.module import Module


class AFTLocalAutoregressive(Module):
    def __init__(self, d_model: int, seq_len: int, s: int, bias: bool = True):
        """
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        """

        super().__init__()

        self.s = s
        # These transform the `query`, `key` and `value` vectors.
        self.query = nn.Linear(d_model, d_model, bias=bias)
        self.key = nn.Linear(d_model, d_model, bias=bias)
        self.value = nn.Linear(d_model, d_model, bias=bias)
        self.weights = nn.Parameter(torch.zeros(seq_len, seq_len), requires_grad=True)
        self.activation = nn.Sigmoid()
        # Output layer
        self.output = nn.Linear(d_model, d_model)

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        `query`, `key` and `value` are the tensors that store
        collection of *query*, *key* and *value* vectors.
        They have shape `[seq_len, batch_size, d_model]`.

        `mask` should be None
        We keep the parameter so that we can use this as an drop in replacement for MHA.
        """

        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        seq_len, batch_size, _ = query.shape

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        max_logit = torch.max(key.max(dim=0)[0],
                              (key + self.weights[:seq_len, :seq_len].max(dim=0)[0].view(-1, 1, 1)).max(dim=0)[0]
                              )[0]
        num = key.new_zeros(key.shape[1:])
        den = key.new_zeros(key.shape[1:])
        x = key.new_zeros(key.shape)
        for t in range(seq_len):
            f = t - self.s + 1
            if f >= 1:
                exp_l = torch.exp(key[f - 1] - max_logit)
                num = num + exp_l * value[f - 1]
                den = den + exp_l
            f = max(0, f)
            exp_l = torch.exp(key[f: t + 1] + self.weights[t, f: t + 1].view(-1, 1, 1) - max_logit.squeeze(0))
            n = num + (exp_l * value[f: t + 1]).sum(dim=0)
            d = den + exp_l.sum(dim=0)
            x[t] = self.activation(query[t]) * n / d

        # Output layer
        return self.output(x)
