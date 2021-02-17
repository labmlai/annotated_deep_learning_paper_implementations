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
    def __init__(self, compression_ratio: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=compression_ratio, stride=compression_ratio)

    def forward(self, mem: torch.Tensor):
        """
        * `mem` has shape `[seq_len, batch, d_model]`
        """

        # Change the dimensions of `mem` so that we can run it through the convolution layer.
        # The convolution layer accepts in the form `[batch, features, sequence]`
        mem = mem.permute(1, 2, 0)
        # Get compressed memory
        c_mem = self.conv(mem)
        # Permute back to form `[seq_len, batch, d_model]`
        return c_mem.permute(2, 0, 1)


class CompressiveTransformerLayer(Module):
    def __init__(self, *,
                 d_model: int,
                 self_attn: RelativeMultiHeadAttention,
                 feed_forward: FeedForward,
                 dropout_prob: float,
                 compress: Conv1dCompression):
        """
        * `d_model` is the token embedding size
        * `self_attn` is the [self attention module](relative_mha.html)
        * `feed_forward` is the feed forward module
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        """
        super().__init__()
        self.compress = compress
        self.size = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def with_memory(self, z: torch.Tensor, mem: Optional[torch.Tensor], c_mem: Optional[torch.Tensor]):
        if mem is None:
            return z

        if c_mem is not None:
            mem = torch.cat((c_mem, mem), dim=0)

        mem = self.norm_self_attn(mem)
        return torch.cat((mem, z), dim=0)

    def forward(self, *,
                x: torch.Tensor,
                mem: Optional[torch.Tensor],
                c_mem: Optional[torch.Tensor],
                mask: torch.Tensor):
        """
        * `x` are the token level feature vectors of shape `[seq_len, batch_size, d_model]`
        * `mem` are the past token level feature vectors of shape `[mem_len + c_mem_len * c, batch_size, d_model]`
        * `mask` is a matrix of shape `[seq_len, c_mem_len + mem_len + seq_len, batch_size]` or `[seq_len, c_mem_len + mem_len + seq_len, 1]`.
        `mask[i, j]` is  true if token at `i` can see token at `j`.
        """

        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        m_z = self.with_memory(z, mem, c_mem)
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
    ## Transformer XL Model

    This consists of multiple transformer XL layers
    """

    def __init__(self, layer: CompressiveTransformerLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, mem: List[torch.Tensor], c_mem: List[torch.Tensor], mask: torch.Tensor):
        """
        * `x` are the token embeddings vectors of shape `[seq_len, batch_size, d_model]`
        * `mem` are the past token level feature vectors of shape `[mem_len, batch_size, d_model]`  for each layer
        * `mask` is the masking matrix
        """
        # List to store token level feature vectors,
        # which will be the memories for the next sequential batch.
        new_mem = []
        # Run through each transformer layer
        for i, layer in enumerate(self.layers):
            # Add to the list of feature vectors
            new_mem.append(x.detach())
            # Memory
            m = mem[i] if mem else None
            # Memory
            cm = c_mem[i] if c_mem else None
            # Run through the transformer XL layer
            x = layer(x=x, mem=m, c_mem=cm, mask=mask)
        # Finally, normalize the vectors
        return self.norm(x), new_mem


class AttentionReconstructionLoss:
    def __init__(self, layers: TypedModuleList[CompressiveTransformerLayer]):
        self.layers = layers
        self.loss_func = nn.MSELoss()

    def prepare_for_attn(self, pmha: PrepareForMultiHeadAttention, x: torch.Tensor):
        head_shape = x.shape[:-1]

        # Linear transform
        weight = pmha.linear.weight.detach()
        bias = pmha.linear.bias.detach() if pmha.linear.bias is not None else None
        x = F.linear(x, weight, bias)

        # Split last dimension into heads
        x = x.view(*head_shape, pmha.heads, pmha.d_k)

        # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, d_model]`
        return x

    def attn(self, layer: RelativeMultiHeadAttention, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
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

    def calc_loss(self, layer: CompressiveTransformerLayer, h: torch.Tensor, mem: torch.Tensor):
        h = h.detach()
        mem = mem.detach()

        c_mem = layer.compress(mem)

        return self.loss_func(self.attn(layer.self_attn, h, mem, mem),
                              self.attn(layer.self_attn, h, c_mem, c_mem))

    def __call__(self, h: List[torch.Tensor], mem: List[torch.Tensor]):
        losses = [self.calc_loss(layer, h[n], mem[n]) for n, layer in enumerate(self.layers)]
        return sum(losses)
