"""

---
title: Receptance Weighted Key Value (RWKV)
summary: >
  This implements the RWKV model 
  using PyTorch with explanations.
---

# Receptance Weighted Key Value (RWKV)

This is a tutorial/implementation of RWKV
from paper [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/pdf/2305.13048.pdf)
in [PyTorch](https://pytorch.org/).

Full definition of a RWKV Language Model, all of it in this single file.
References:
1) [the official RWKV PyTorch implementation released by Bo Peng](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/src/model.py)
2) [huggingface/transformers PyTorch implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from labml_helpers.module import Module

PREV_X_TIME = 0
NUM_STATE = 1
DEN_STATE = 2
MAX_STATE = 3
PREV_X_CHANNEL = 4


class LayerNorm(Module):
    """
    ### Layer normalization with bias
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class L2Wrap(torch.autograd.Function):
    """
    ### L2 loss wrapper

    [ref](https://github.com/BlinkDL/RWKV-LM/blob/cca1b5e8e597cf40675882bb10b46287c844e35c/RWKV-v4/src/model.py#L21)
    """

    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return grad_output, gy


class ChannelMixing(Module):
    """
    ### Channel Mixing
    """

    def __init__(self, config, layer_id):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        # token shifting
        self.layer_id = layer_id

        n_embd = config.n_embd
        intermediate_size = (
            config.intermediate_size if config.intermediate_size is not None else 4 * n_embd
        )

        # Learnable Matrix
        self.key_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.value_proj = nn.Linear(intermediate_size, n_embd, bias=False)
        self.receptance_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Learnable Vector
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, n_embd))

    def forward(self, x, state=None):
        """
        # x = (Batch,Time,Channel)
        """
        if state is not None:
            prev_x = state[self.layer_id, :, [PREV_X_CHANNEL], :]
            state[self.layer_id, :, [PREV_X_CHANNEL], :] = x
        else:
            prev_x = self.time_shift(x)

        # $r_t=W_r \cdot (\mu_r x_t + (1-\mu_r)x_{t-1})$
        receptance = x * self.time_mix_receptance + prev_x * (1 - self.time_mix_receptance)
        receptance = self.receptance_proj(receptance)

        # $k_t=W_k \cdot (\mu_k x_t + (1-\mu_k)x_{t-1})$
        key = x * self.time_mix_key + prev_x * (1 - self.time_mix_key)
        key = self.key_proj(key)

        # $V_t=W_v \cdot max(k_t,0)^2$
        value = self.value_proj(torch.square(torch.relu(key)))

        # $o_t=\sigma(r_t) \odot v_t$
        out = F.sigmoid(receptance) * value
        return out, state


class TimeMixing(Module):
    """
    ### Time Mixing
    """

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.layer_id = layer_id

        n_embd = config.n_embd
        attn_sz = n_embd

        # learnable matrix
        self.key_proj = nn.Linear(n_embd, attn_sz, bias=False)
        self.value_proj = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance_proj = nn.Linear(n_embd, attn_sz, bias=False)
        self.output_proj = nn.Linear(attn_sz, n_embd, bias=False)

        # learnable vector
        self.time_decay = nn.Parameter(torch.empty(attn_sz))
        self.time_first = nn.Parameter(torch.empty(attn_sz))
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_value = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, n_embd))

    def forward(self, x, state=None):
        """
        x = (Batch,Time,Channel)
        """
        if state is not None:
            prev_x = state[self.layer_id, :, [PREV_X_TIME], :]
            state[self.layer_id, :, [PREV_X_TIME], :] = x
        else:
            prev_x = self.time_shift(x)

        # $r_t=W_r \cdot (\mu_r x_t + (1-\mu_r)x_{t-1})$
        receptance = x * self.time_mix_receptance + prev_x * (1 - self.time_mix_receptance)
        receptance = self.receptance_proj(receptance)

        # $k_t=W_k \cdot (\mu_k x_t + (1-\mu_k)x_{t-1})$
        key = x * self.time_mix_key + prev_x * (1 - self.time_mix_key)
        key = self.key_proj(key)

        # $v_t=W_v \cdot (\mu_v x_t + (1-\mu_v)x_{t-1})$
        value = x * self.time_mix_value + prev_x * (1 - self.time_mix_value)
        value = self.value_proj(value)

        # WKV calculation
        _, seq_length, _ = key.size()
        output = torch.zeros_like(key)

        if state is None:
            num_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
            den_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
            max_state = torch.zeros_like(key[:, 0], dtype=torch.float32) - 1e38
        else:
            num_state = state[self.layer_id, :, NUM_STATE, :]
            den_state = state[self.layer_id, :, DEN_STATE, :]
            max_state = state[self.layer_id, :, MAX_STATE, :]

        time_decay = -torch.exp(self.time_decay)

        for current_index in range(seq_length):
            current_key = key[:, current_index].float()
            current_value = value[:, current_index]

            # $wkv_t=\frac{\sum^{t-1}_{i=1}d^{-(t-1-i)w+k_i}v_i+e^{u+k_t}v_t}{\sum^{t-1}_{i=1}e^{-(t-1-i)w+k_i}+e^{u+k_t}}$
            max_for_output = torch.maximum(max_state, current_key + self.time_first)
            e1 = torch.exp(max_state - max_for_output)
            e2 = torch.exp(current_key + self.time_first - max_for_output)
            numerator = e1 * num_state + e2 * current_value
            denominator = e1 * den_state + e2
            output[:, current_index] = (numerator / denominator).to(output.dtype)

            # Update state for next iteration
            max_for_state = torch.maximum(max_state + time_decay, current_key)
            e1 = torch.exp(max_state + time_decay - max_for_state)
            e2 = torch.exp(current_key - max_for_state)
            num_state = e1 * num_state + e2 * current_value
            den_state = e1 * den_state + e2
            max_state = max_for_state

        # update states
        state[self.layer_id, :, NUM_STATE, :] = num_state
        state[self.layer_id, :, DEN_STATE, :] = den_state
        state[self.layer_id, :, MAX_STATE, :] = max_state
        wkv, state = self.wkv_function(key, value, use_customized_cuda_kernel=self.config.use_customized_cuda_kernel,
                                       state=state)

        # $o_t=W_o \cdot (\sigma(r_t) \odot wkv_t)$
        rwkv = F.sigmoid(receptance) * wkv
        rwkv = self.output_proj(rwkv)

        return rwkv, state


class Block(Module):
    """
    ## RWKV block element
    """

    def __init__(self, config, layer_id):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = TimeMixing(config, layer_id)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ffn = ChannelMixing(config, layer_id)

    def forward(self, x, state=None):
        # state: [batch_size, 5 , n_embd]

        # time mixing
        residual = x
        x, state = self.attn(self.ln_1(x), state=state)
        x = x + residual

        # channel mixing
        residual = x
        x, state = self.ffn(self.ln_2(x), state=state)
        x = x + residual
        return x, state


class RWKV(Module):
    """
    ## RWKV
    """
    def __init__(self, config, lr_init=0.0008):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.lr_init = lr_init  ## used to initialize embedding parameters
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd

        # Initiate model layers
        self.rwkv = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            ln_p=LayerNorm(config.n_embd, bias=config.bias),
            h=nn.ModuleList([Block(config, layer_id) for layer_id in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Output linear layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None, state=None, return_state=False):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Embedding Layer
        x = self.rwkv.wte(idx)

        # Layer Norm
        x = self.rwkv.ln_p(x)

        # RWKV Blocks
        for block_idx, block in enumerate(self.rwkv.h):
            x, state = block(x, state)
        x = self.rwkv.ln_f(x)

        # Logit Layer and loss Function (for training)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = L2Wrap.apply(loss, logits)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        # Return Logits and loss
        if return_state:
            return logits, loss, state
        else:
            return logits, loss
