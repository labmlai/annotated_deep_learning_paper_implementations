from typing import List, Optional

import torch
import torch.nn as nn

from labml_helpers.module import Module
from labml_nn.utils import clone_module_list
from .relative_mha import RelativeMultiHeadAttention
from ..feed_forward import FeedForward


class TransformerXLLayer(Module):
    def __init__(self, *,
                 d_model: int,
                 self_attn: RelativeMultiHeadAttention,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        """
        * `d_model` is the token embedding size
        * `self_attn` is the self attention module
        * `feed_forward` is the feed forward module
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        """
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])
        # Whether to save input to the feed forward layer
        self.is_save_ff_input = False

    def forward(self, *,
                x: torch.Tensor,
                mem: Optional[torch.Tensor],
                mask: torch.Tensor):
        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        # Normalize memory
        if mem is not None:
            mem = self.norm_self_attn(mem)
            zm = torch.cat((mem, z), dim=0)
        else:
            zm = z
        # Run through self attention, i.e. keys and values are from self
        self_attn = self.self_attn(query=z, key=zm, value=zm, mask=mask)
        # Add the self attention results
        x = x + self.dropout(self_attn)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Save the input to the feed forward layer if specified
        if self.is_save_ff_input:
            self.ff_input = z.clone()
        # Pass through the feed-forward network
        ff = self.feed_forward(z)
        # Add the feed-forward results back
        x = x + self.dropout(ff)

        return x


class TransformerXL(Module):
    def __init__(self, layer: TransformerXLLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, mem: List[torch.Tensor], mask: torch.Tensor):
        new_mem = []
        # Run through each transformer layer
        for i, layer in enumerate(self.layers):
            new_mem.append(x.detach())
            m = mem[i] if mem else None
            x = layer(x=x, mem=m, mask=mask)
        # Finally, normalize the vectors
        return self.norm(x), new_mem
