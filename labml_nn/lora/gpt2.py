"""
---
title: GPT-2 with LoRA
summary: GPT-2 implementation with LoRA modules
---

# GPT-2 with [LoRA modules](index.html)

Here's [the training code](experiment.html) for training a GPT2 model with LoRA
 on Tiny Shakespeare dataset.
"""

import torch
import torch.nn as nn

from labml_nn.lora import Linear, Embedding


class FFN(nn.Module):
    """
    ### Feedforward Network
    """

    def __init__(self, d_model: int, d_ff: int, r: int):
        """
        :param d_model: is the number of dimensions
        :param d_ff: is the size of the hidden dimension
        :param r: is the lora rank
        """
        super().__init__()

        # The linear layers and the activation
        self.linear_in = Linear(d_model, d_ff, r=r, bias=True)
        self.linear_out = Linear(d_ff, d_model, r=r, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: is the embeddings tensor with shape `[batch_size, seq_len, d_model]`
        """
        x = self.linear_in(x)
        x = self.act(x)
        x = self.linear_out(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    ### Multi-Head Attention
    """

    def __init__(self, d_model: int, n_heads: int, r: int):
        """
        :param d_model: is the number of dimensions in the embeddings
        :param n_heads: is the number of heads
        :param r: is the lora rank
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Linear transformation for QKV
        self.qkv_projection = Linear(d_model, d_model * 3, r=r, bias=True)
        # Output projection
        self.output_projection = Linear(d_model, d_model, r=r, bias=True)

    def _split_heads(self, x: torch.Tensor):
        """
        :param x: is the tensor with shape `[batch_size, seq_len, d_model]`
        """
        # Split last dimension to `[n_heads, d_head]`
        x = x.view(x.shape[:-1] + (self.n_heads, self.d_head))
        # Reorder to `[batch_size, head, seq_length, d_head]`
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: is the embeddings tensor with shape `[batch_size, seq_len, d_model]`
        """
        batch_size, seq_length, _ = x.shape

        # Get query, key and value
        q, k, v = self.qkv_projection(x).split(self.d_model, dim=-1)

        # Transform them from shape  `[batch_size, seq_len, d_model]` to `[batch_size, head, seq_length, d_head]`
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Apply causal attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Transform them from shape  `[batch_size, head, seq_length, d_head]` to `[batch_size, seq_len, d_model]`
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.d_model)

        # Final project
        return self.output_projection(attn_output)


class Block(nn.Module):
    """
    ### Decoder block
    """

    def __init__(self, d_model: int, n_heads: int, layer_norm_epsilon: float, r: int):
        """
        :param d_model: is the number of dimensions in the embeddings
        :param n_heads: is the number of heads
        :param layer_norm_epsilon: is the layer norm epsilon
        :param r: is the lora rank
        """
        super().__init__()
        # Attention pre-normalization layer
        self.attn_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        # Attention layer
        self.attn = MultiHeadAttention(d_model, n_heads, r)
        # FFN pre-normalization layer
        self.ffn_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        # Feed-forward network
        self.ffn = FFN(d_model, d_model * 4, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: is the embeddings tensor with shape `[batch_size, seq_len, d_model]`
        """
        # Attention
        x = x + self.attn(self.attn_norm(x))
        # FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x


class GPTModel(nn.Module):
    """
    ## GPT2 Model
    """

    def __init__(self, *, d_model: int,
                 n_heads: int, n_layers: int,
                 n_positions: int,
                 layer_norm_epsilon: float,
                 vocab_size: int, r: int):
        """
        :param d_model: is the number of dimensions in the embeddings
        :param n_heads: is the number of attention heads
        :param n_layers: is the number of decoder layers
        :param n_positions: is the number of positional embeddings
        :param layer_norm_epsilon: is the layer norm epsilon
        :param vocab_size: is the vocabulary size
        :param r: is the lora rank
        """
        super().__init__()

        # Token and absolute positional embeddings
        self.token_embedding = Embedding(vocab_size, d_model, r=r)
        self.position_embedding = Embedding(n_positions, d_model, r=r)

        # Decoder blocks
        self.blocks = nn.ModuleList([Block(d_model, n_heads, layer_norm_epsilon, r=r)
                                     for _ in range(n_layers)])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        # Projection layer to logit space
        self.lm_head = Linear(d_model, vocab_size, r=r, bias=False)

    def forward(self, input_ids: torch.Tensor):
        """
        :param input_ids: has shape `[batch_size, seq_len]`
        """
        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        token_embeddings = self.token_embedding(input_ids)
        # Get position ids
        position_ids = torch.arange(seq_len, device=input_ids.device)[None, :]
        # Get position embeddings
        position_embeddings = self.position_embedding(position_ids)

        # Add position embeddings
        x = token_embeddings + position_embeddings

        # Run through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.final_norm(x)
        # Get logits from projection layer
        return self.lm_head(x)
