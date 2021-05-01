"""
---
title: Configurable Transformer Components
summary: These are configurable components that can be re-used quite easily.
---

# Configurable Transformer Components
"""
import copy

import torch.nn as nn

from labml.configs import BaseConfigs, option, calculate, aggregate
from labml_helpers.module import Module
from .feed_forward import FeedForward
from .mha import MultiHeadAttention
from .models import EmbeddingsWithPositionalEncoding, EmbeddingsWithLearnedPositionalEncoding, TransformerLayer, \
    Encoder, Decoder, Generator, EncoderDecoder


class FeedForwardConfigs(BaseConfigs):
    """
    <a id="FFN">
    ## FFN Configurations
    </a>

    Creates a Position-wise FeedForward Network defined in
    [`feed_forward.py`](feed_forward.html).
    """
    # Position-wise feedforward layer
    ffn: FeedForward
    # Number of features in the embedding
    d_model: int
    # Number of features in in the hidden layer
    d_ff: int = 2048
    # Dropout probability
    dropout: float = 0.1
    # Activation in position-wise feedforward layer
    activation: nn.Module = 'ReLU'
    # Whether the FFN layer should be gated
    is_gated: bool = False
    # Whether the first fully connected layer should have a learnable bias
    bias1: bool = True
    # Whether the second fully connected layer should have a learnable bias
    bias2: bool = True
    # Whether the fully connected layer for the gate should have a learnable bias
    bias_gate: bool = False
    # Predefined GLU variants
    glu_variant: str = 'none'


@option(FeedForwardConfigs.activation, 'ReLU')
def _ffn_activation_relu():
    """
    ### ReLU activation

    $$\max(0, x)$$
    """
    return nn.ReLU()


@option(FeedForwardConfigs.activation, 'GELU')
def _ffn_activation_gelu():
    """
    ### GELU activation

    $$x \Phi(x)$$ where $\Phi(x) = P(X \le x), X \sim \mathcal{N}(0,1)$

    It was introduced in paper [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415).
    """
    return nn.GELU()


@option(FeedForwardConfigs.ffn, 'default')
def _feed_forward(c: FeedForwardConfigs):
    """
    Initialize a [feed forward network](feed_forward.html)
    """
    return FeedForward(c.d_model, c.d_ff,
                       dropout=c.dropout,
                       activation=c.activation,
                       is_gated=c.is_gated,
                       bias1=c.bias1,
                       bias2=c.bias2,
                       bias_gate=c.bias_gate)

# ## GLU Variants
# These are variants with gated hidden layers for the FFN
# as introduced in paper [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).
# We have omitted the bias terms as specified in the paper.

# ### FFN with Gated Linear Units
#
# $$FFN_{GLU}(x)(x, W_1, V, W_2) = (\sigma(x W_1) \otimes x V) W_2$$
aggregate(FeedForwardConfigs.glu_variant, 'GLU',
          (FeedForwardConfigs.is_gated, True),
          (FeedForwardConfigs.bias1, False),
          (FeedForwardConfigs.bias2, False),
          (FeedForwardConfigs.bias_gate, False),
          (FeedForwardConfigs.activation, nn.Sigmoid()))

# ### FFN with Bilinear hidden layer
#
# $$FFN_{Bilinear}(x)(x, W_1, V, W_2) = (x W_1 \otimes x V) W_2$$
aggregate(FeedForwardConfigs.glu_variant, 'Bilinear',
          (FeedForwardConfigs.is_gated, True),
          (FeedForwardConfigs.bias1, False),
          (FeedForwardConfigs.bias2, False),
          (FeedForwardConfigs.bias_gate, False),
          (FeedForwardConfigs.activation, nn.Identity()))

# ### FFN with ReLU gate
#
# $$FFN_{ReGLU}(x)(x, W_1, V, W_2) = (\max(0, x W_1) \otimes x V) W_2$$
aggregate(FeedForwardConfigs.glu_variant, 'ReGLU',
          (FeedForwardConfigs.is_gated, True),
          (FeedForwardConfigs.bias1, False),
          (FeedForwardConfigs.bias2, False),
          (FeedForwardConfigs.bias_gate, False),
          (FeedForwardConfigs.activation, nn.ReLU()))

# ### FFN with GELU gate
#
# $$FFN_{GEGLU}(x)(x, W_1, V, W_2) = (\text{GELU}(x W_1) \otimes x V) W_2$$
aggregate(FeedForwardConfigs.glu_variant, 'GEGLU',
          (FeedForwardConfigs.is_gated, True),
          (FeedForwardConfigs.bias1, False),
          (FeedForwardConfigs.bias2, False),
          (FeedForwardConfigs.bias_gate, False),
          (FeedForwardConfigs.activation, nn.GELU()))

# ### FFN with Swish gate
#
# $$FFN_{SwiGLU}(x)(x, W_1, V, W_2) = (\text{Swish}_1(x W_1) \otimes x V) W_2$$
# where $\text{Swish}_\beta(x) = x \sigma(\beta x)$
aggregate(FeedForwardConfigs.glu_variant, 'SwiGLU',
          (FeedForwardConfigs.is_gated, True),
          (FeedForwardConfigs.bias1, False),
          (FeedForwardConfigs.bias2, False),
          (FeedForwardConfigs.bias_gate, False),
          (FeedForwardConfigs.activation, nn.SiLU()))


class TransformerConfigs(BaseConfigs):
    """
    <a id="TransformerConfigs">
    ## Transformer Configurations
    </a>
    
    This defines configurations for a transformer.
    The configurations are calculate using option functions.
    These are lazy loaded and therefore only the necessary modules
    are calculated.
    """
    # Number of attention heads
    n_heads: int = 8
    # Transformer embedding size
    d_model: int = 512
    # Number of layers
    n_layers: int = 6
    # Dropout probability
    dropout: float = 0.1
    # Number of tokens in the source vocabulary (for token embeddings)
    n_src_vocab: int
    # Number of tokens in the target vocabulary (to generate logits for prediction)
    n_tgt_vocab: int

    # The encoder self attention
    encoder_attn: MultiHeadAttention = 'mha'
    # The decoder self attention
    decoder_attn: MultiHeadAttention = 'mha'
    # The decoder memory attention
    decoder_mem_attn: MultiHeadAttention = 'mha'

    # Configurable Feedforward Layer
    ffn: FeedForwardConfigs

    # Encoder layer
    encoder_layer: TransformerLayer = 'default'
    # Decoder layer
    decoder_layer: TransformerLayer = 'default'

    # Encoder consisting of multiple encoder layers
    encoder: Encoder = 'default'
    # Encoder consisting of multiple decoder layers
    decoder: Decoder = 'default'

    # Embedding layer for source
    src_embed: Module = 'fixed_pos'
    # Embedding layer for target (for decoder)
    tgt_embed: Module = 'fixed_pos'

    # Logit generator for prediction
    generator: Generator = 'default'

    # Encoder-decoder
    encoder_decoder: EncoderDecoder


# ### Multi-head Attention
def _mha(c: TransformerConfigs):
    return MultiHeadAttention(c.n_heads, c.d_model)


calculate(TransformerConfigs.encoder_attn, 'mha', _mha)
calculate(TransformerConfigs.decoder_attn, 'mha', _mha)
calculate(TransformerConfigs.decoder_mem_attn, 'mha', _mha)


# ### Relative Multi-head Attention
def _relative_mha(c: TransformerConfigs):
    from labml_nn.transformers.xl.relative_mha import RelativeMultiHeadAttention
    return RelativeMultiHeadAttention(c.n_heads, c.d_model)


calculate(TransformerConfigs.encoder_attn, 'relative', _relative_mha)
calculate(TransformerConfigs.decoder_attn, 'relative', _relative_mha)
calculate(TransformerConfigs.decoder_mem_attn, 'relative', _relative_mha)


@option(TransformerConfigs.ffn, 'default')
def _feed_forward(c: TransformerConfigs):
    """
    Create feedforward layer configurations
    """
    conf = FeedForwardConfigs()
    conf.set_default(FeedForwardConfigs.d_model, func=lambda: c.d_model)
    conf.set_default(FeedForwardConfigs.dropout, func=lambda: c.dropout)
    return conf


@option(TransformerConfigs.encoder_layer, 'default')
def _encoder_layer(c: TransformerConfigs):
    """
    Encoder layer
    """
    return TransformerLayer(d_model=c.d_model, self_attn=c.encoder_attn,
                            src_attn=None, feed_forward=copy.deepcopy(c.ffn.ffn),
                            dropout_prob=c.dropout)


@option(TransformerConfigs.decoder_layer, 'default')
def _decoder_layer(c: TransformerConfigs):
    """
    Decoder layer
    """
    return TransformerLayer(d_model=c.d_model, self_attn=c.decoder_attn,
                            src_attn=c.decoder_mem_attn, feed_forward=copy.deepcopy(c.ffn.ffn),
                            dropout_prob=c.dropout)


@option(TransformerConfigs.encoder, 'default')
def _encoder(c: TransformerConfigs):
    """
    Encoder
    """
    return Encoder(c.encoder_layer, c.n_layers)


@option(TransformerConfigs.decoder, 'default')
def _decoder(c: TransformerConfigs):
    """
    Decoder
    """
    return Decoder(c.decoder_layer, c.n_layers)


@option(TransformerConfigs.generator, 'default')
def _generator(c: TransformerConfigs):
    """
    Logit generator
    """
    return Generator(c.n_tgt_vocab, c.d_model)


# ### Fixed Positional Embeddings
@option(TransformerConfigs.src_embed, 'fixed_pos')
def _src_embed_with_positional(c: TransformerConfigs):
    """
    Source embedding with fixed positional encodings
    """
    return EmbeddingsWithPositionalEncoding(c.d_model, c.n_src_vocab)


@option(TransformerConfigs.tgt_embed, 'fixed_pos')
def _tgt_embed_with_positional(c: TransformerConfigs):
    """
    Target embedding with fixed positional encodings
    """
    return EmbeddingsWithPositionalEncoding(c.d_model, c.n_tgt_vocab)


# ### Learned Positional Embeddings
@option(TransformerConfigs.src_embed, 'learned_pos')
def _src_embed_with_learned_positional(c: TransformerConfigs):
    """
    Source embedding with learned positional encodings
    """
    return EmbeddingsWithLearnedPositionalEncoding(c.d_model, c.n_src_vocab)


@option(TransformerConfigs.tgt_embed, 'learned_pos')
def _tgt_embed_with_learned_positional(c: TransformerConfigs):
    """
    Target embedding with learned positional encodings
    """
    return EmbeddingsWithLearnedPositionalEncoding(c.d_model, c.n_tgt_vocab)


# ### No Positional Embeddings
@option(TransformerConfigs.src_embed, 'no_pos')
def _src_embed_without_positional(c: TransformerConfigs):
    """
    Source embedding without positional encodings
    """
    return nn.Embedding(c.n_src_vocab, c.d_model)


@option(TransformerConfigs.tgt_embed, 'no_pos')
def _tgt_embed_without_positional(c: TransformerConfigs):
    return nn.Embedding(c.n_tgt_vocab, c.d_model)


@option(TransformerConfigs.encoder_decoder, 'default')
def _encoder_decoder(c: TransformerConfigs):
    return EncoderDecoder(c.encoder, c.decoder, c.src_embed, c.tgt_embed, c.generator)
