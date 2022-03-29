import copy

import torch
import torch.nn as nn

from labml_helpers.module import Module
from labml_nn.normalization.deep_norm import DeepNorm, deep_norm_init
from labml_nn.transformers import MultiHeadAttention
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.utils import subsequent_mask


class TransformerLayer(nn.Module):
    def __init__(self, *,
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 feed_forward: FeedForward,
                 deep_norm_alpha: float,
                 deep_norm_beta: float,
                 ):
        """
        * `d_model` is the token embedding size
        * `self_attn` is the self attention module
        * `feed_forward` is the feed forward module
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        """
        super().__init__()
        self.size = d_model
        self.self_attn = DeepNorm(self_attn, deep_norm_alpha, [d_model])
        self.feed_forward = DeepNorm(feed_forward, deep_norm_alpha, [d_model])

        deep_norm_init(feed_forward.layer1.weight, deep_norm_beta)
        deep_norm_init(feed_forward.layer2.weight, deep_norm_beta)

        deep_norm_init(self_attn.value.linear.weight, deep_norm_beta)
        deep_norm_init(self_attn.output.weight, deep_norm_beta)

    def forward(self, *,
                x: torch.Tensor,
                mask: torch.Tensor):
        # Run through self attention, i.e. keys and values are from self
        x = self.self_attn(query=x, key=x, value=x, mask=mask)
        # Pass through the feed-forward network
        x = self.feed_forward(x)

        return x


class AutoregressiveTransformer(Module):
    """
    ## Auto-Regressive model
    """

    def __init__(self, n_tokens: int, d_model: int, n_layers: int, layer: TransformerLayer):
        """
        * `encoder` is the transformer [Encoder](../models.html#Encoder)
        * `src_embed` is the token
        [embedding module (with positional encodings)](../models.html#EmbeddingsWithLearnedPositionalEncoding)
        * `generator` is the [final fully connected layer](../models.html#Generator) that gives the logits.
        """
        super().__init__()
        self.encoder = nn.Sequential(*[copy.deepcopy(layer) for _ in range(n_layers)])

        self.emb = nn.Embedding(n_tokens, d_model)
        self.readout = nn.Linear(d_model, n_tokens)

        # The mask will be initialized on the first call
        self.mask = None

    def forward(self, x: torch.Tensor):
        # Create subsequent mask if mask is not initialized
        # or if the size of the mask is different
        if self.mask is None or self.mask.size(0) != len(x):
            # Subsequent mask, will mask out tokens from seeing future tokens
            self.mask = subsequent_mask(len(x)).to(x.device)
        # Get the token embeddings with positional encodings
        x = self.emb(x)
        # Transformer encoder
        x = self.encoder(x, self.mask)
        # Get logits
        x = self.readout(x)

        # Return results
        return x


def _test():
    AutoregressiveTransformer(65, 32, 3,
                              TransformerLayer(d_model=32,
                                               deep_norm_alpha=0.3,
                                               deep_norm_beta=0.3,
                                               feed_forward=FeedForward(d_model=32, d_ff=32 * 4),
                                               self_attn=MultiHeadAttention(4, 32)))


if __name__ == '__main__':
    _test()
