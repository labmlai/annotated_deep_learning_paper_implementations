"""
---
title: Compressive Transformer
summary: >
  Documented implementation with explanations of a
  Compressive Transformer model.
---

# Compressive Transformer

This is an implementation of
[Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/abs/1911.05507)
in [PyTorch](https://pytorch.org).

This is an extension of [Transformer XL](../xl/index.html) where past memories
are compressed to give a longer attention range.
That is, the furthest $n_{cm} c$ memories are compressed into
$n_{cm}$ memories, where $c$ is the compression rate.

## Compression operation

The compression operation is defined as
$f_c: \mathbb{R}^{nc \times d} \rightarrow \mathbb{R}^{n \times d}$.
The paper introduces multiple choices for $f_c$ and we have only implemented
1D convolution which seems to give best results.
Each layer has a separate compression operation $f_c^{(i)}$ where
$i$ is the layer number.

## Training compression operation

Since training compression with BPTT requires maintaining
a very large computational graph (many time steps), paper proposes
an *auto-encoding loss* and an *attention reconstruction loss*.
The auto-encoding loss, decodes the original memories from the compressed memories,
and calculate the loss.
Attention reconstruction loss computes the multi-headed attention results
on the compressed memory and on uncompressed memory and get a mean squared error
between them.
We have implemented the latter here since it gives better results.

This implementation uses pre-layer norm while the paper uses post-layer norm.
Pre-layer norm does the layer norm before FFN[../feedforward.html) and
self attention, and the pass through in the residual connection is not normalized.
This is supposed to be more stable in standard transformer setups.

Here's [the training code](experiment.html) and a notebook for training a compressive transformer
model on Tiny Shakespeare dataset.
"""

from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn

from labml_helpers.module import Module, TypedModuleList
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
from labml_nn.transformers.xl.relative_mha import RelativeMultiHeadAttention
from labml_nn.utils import clone_module_list


class Conv1dCompression(Module):
    """
    ## 1D Convolution Compression $f_c$

    This is a simple wrapper around
    [`nn.Conv1d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
    with some tensor dimension permutations.
    """
    def __init__(self, compression_rate: int, d_model: int):
        """
        * `compression_rate` $c$
        * `d_model` is the embedding size
        """
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=compression_rate, stride=compression_rate)

    def forward(self, mem: torch.Tensor):
        """
        `mem` has shape `[seq_len, batch, d_model]`
        """

        # Permute the dimensions of `mem` so that we can run it through the convolution layer.
        # The convolution layer accepts in the form `[batch, features, sequence]`
        mem = mem.permute(1, 2, 0)
        # Get compressed memory by running it through the convolution layer
        c_mem = self.conv(mem)
        # Permute back to form `[seq_len, batch, d_model]`
        return c_mem.permute(2, 0, 1)


class CompressiveTransformerLayer(Module):
    """
    ## Compressive Transformer Layer

    This is the implementation of a single compressive transformer layer
    """
    def __init__(self, *,
                 d_model: int,
                 self_attn: RelativeMultiHeadAttention,
                 feed_forward: FeedForward,
                 dropout_prob: float,
                 compress: Conv1dCompression):
        """
        * `d_model` is the token embedding size
        * `self_attn` is the [self attention module](../xl/relative_mha.html)
        * `feed_forward` is the [feed forward module](../feed_forward.html)
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        * `compress` is the compression function $f_c$
        """
        super().__init__()
        self.compress = compress
        self.size = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def concat_memory(self, z: torch.Tensor, mem: Optional[torch.Tensor], c_mem: Optional[torch.Tensor]):
        """
        Concatenate the normalized token embeddings with memory and compressed memory.

        * `z` is layer normalized token embeddings.
        * `mem` and `c_mem` are memory and compressed memory (not normalized).
        """

        # If there is no memory just return the token embeddings
        if mem is None:
            return z

        # If there are compressed memory concatenate that with memory
        if c_mem is not None:
            mem = torch.cat((c_mem, mem), dim=0)

        # Run the memory through the normalization layer
        mem = self.norm_self_attn(mem)
        # Concatenate normalized memory and normalized token embeddings
        return torch.cat((mem, z), dim=0)

    def forward(self, *,
                x: torch.Tensor,
                mem: Optional[torch.Tensor],
                c_mem: Optional[torch.Tensor],
                mask: torch.Tensor):
        """
        * `x` is a tensor of token level feature vectors of shape `[seq_len, batch_size, d_model]`
        * `mem` is a tensor of the past token level feature vectors (memory) of shape `[mem_len, batch_size, d_model]`
        * `c_mem` is a tensor of the compressed memory `[c_mem_len, batch_size, d_model]`
        * `mask` is a matrix of shape `[seq_len, c_mem_len + mem_len + seq_len, batch_size]` or `[seq_len, c_mem_len + mem_len + seq_len, 1]`.
        `mask[i, j]` is  true if token at `i` can see token at `j`.
        """

        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        # Normalize and concatenate memory and compressed memory
        m_z = self.concat_memory(z, mem, c_mem)
        # Attention
        self_attn = self.self_attn(query=z, key=m_z, value=m_z, mask=mask)
        # Add the attention results
        x = x + self.dropout(self_attn)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the feed-forward network
        ff = self.feed_forward(z)
        # Add the feed-forward results back
        x = x + self.dropout(ff)

        #
        return x


class CompressiveTransformer(Module):
    """
    ## Compressive Transformer Model

    This consists of multiple compressive transformer layers
    """

    def __init__(self, layer: CompressiveTransformerLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, mem: List[torch.Tensor], c_mem: List[torch.Tensor], mask: torch.Tensor):
        """
        * `x` is a tensor of the token embeddings vectors of shape `[seq_len, batch_size, d_model]`
        * `mem` is a list of tensors of the past token level feature vectors of shape
         `[mem_len, batch_size, d_model]` for each layer
        * `c_mem` is a list of tensors of the compressed memory
         `[c_mem_len, batch_size, d_model]` for each layer
        * `mask` is the masking matrix
        """
        # List to store token level feature vectors,
        # which will become the memories for the next sequential batch.
        new_mem = []
        # Run through each transformer layer
        for i, layer in enumerate(self.layers):
            # Add to the list of feature vectors
            new_mem.append(x.detach())
            # Memory
            m = mem[i] if mem else None
            # Compressed Memory
            cm = c_mem[i] if c_mem else None
            # Run through the transformer XL layer
            x = layer(x=x, mem=m, c_mem=cm, mask=mask)
        # Finally, normalize the vectors
        return self.norm(x), new_mem


class AttentionReconstructionLoss:
    """
    ## Attention Reconstruction Loss

    Attention reconstruction loss recreates the self attention output with
    uncompressed memory and with compressed memory and calculate mean squared error
    between the two. It does this without positional encoding.

    When calculating and training the compression function $f_c$ with attention
    reconstruction loss all parameters but $f_c$ are frozen.
    This includes key value projections and bias/scaling after normalization.

    Since this loss can be computed independently of the cross-entropy-loss of the model
    you can have a separate optimizer that only updates $f_c$.
    However, we use the same optimizer to update $f_c$ so when calculating
    attention reconstruction loss we detach all other parameters except $f_c$
    from the gradient computation.
    """
    def __init__(self, layers: TypedModuleList[CompressiveTransformerLayer]):
        """
        `layers` is the list of Compressive Transformer layers
        """
        self.layers = layers
        self.loss_func = nn.MSELoss()

    def prepare_for_attn(self, pmha: PrepareForMultiHeadAttention, x: torch.Tensor):
        """
        This is a reimplementation of ['PrepareForMultiHeadAttention'](../mha.html#PrepareMHA)
        where the projections are done with the parameters detached from gradient computation.

        * `pmha* is the ['PrepareForMultiHeadAttention'](../mha.html#PrepareMHA) module
        * `x` is tensor with the token embeddings
        """

        # Shape of the input except embedding dimension; `[seq_len, batch_size]`.
        head_shape = x.shape[:-1]

        # Detach projection weights and bias
        weight = pmha.linear.weight.detach()
        bias = pmha.linear.bias.detach() if pmha.linear.bias is not None else None
        # Linear transform
        x = F.linear(x, weight, bias)

        # Split last dimension into heads
        x = x.view(*head_shape, pmha.heads, pmha.d_k)

        # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, d_model]`
        return x

    def attn(self, layer: RelativeMultiHeadAttention, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        This is a reimplementation of ['Multi-Head Attention'](../mha.html#MHA) which calls
        `prepare_for_attn` instead of ['PrepareForMultiHeadAttention'](../mha.html#PrepareMHA)
        to detach projection parameters.
        """
        # Calculate query, key and value projections
        query = self.prepare_for_attn(layer.query, query)
        key = self.prepare_for_attn(layer.key, key)
        value = self.prepare_for_attn(layer.value, value)

        # Compute attention scores $Q K^\top$.
        # This gives a tensor of shape `[seq_len, seq_len, batch_size, heads]`.
        scores = torch.einsum('ibhd,jbhd->ijbh', query, key)

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= layer.scale

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = layer.softmax(scores)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
        return torch.einsum("ijbh,jbhd->ibhd", attn, value)

    def norm(self, ln: nn.LayerNorm, x: torch.Tensor):
        """
        Perform layer normalization with shift and scale parameters detached.
        """

        # Detach shift(`bias`) and scaling(`weight`) parameters
        weight = ln.weight.detach() if ln.weight is not None else None
        bias = ln.bias.detach() if ln.bias is not None else None

        # Layer normalization
        return F.layer_norm(x, ln.normalized_shape, weight, bias, ln.eps)

    def calc_loss(self, layer: CompressiveTransformerLayer, h: torch.Tensor, mem: torch.Tensor):
        """
        This calculates the loss for a layer
        """

        # Detach the token embeddings and memory.
        h = h.detach()
        mem = mem.detach()

        # Compress the memory with $f_c^{(i)}$.
        # The parameters of $f_c^{(i)}$ are the only parameters not detached from gradient computation.
        c_mem = layer.compress(mem)

        # Normalize the embeddings and memories
        h = self.norm(layer.norm_self_attn, h)
        mem = self.norm(layer.norm_self_attn, mem)
        c_mem = self.norm(layer.norm_self_attn, c_mem)

        # Calculate attention with uncompressed memory
        attn_mem = self.attn(layer.self_attn, h, mem, mem)
        # Calculate the attention with compressed memory
        attn_cmem = self.attn(layer.self_attn, h, c_mem, c_mem)

        # Calculate the mean square error
        return self.loss_func(attn_cmem, attn_mem)

    def __call__(self, h: List[torch.Tensor], mem: List[torch.Tensor]):
        # Calculate the losses for each layer
        losses = [self.calc_loss(layer, h[n], mem[n]) for n, layer in enumerate(self.layers)]
        # Sum of the losses
        return sum(losses)
