"""
---
title: Configurable Transformer Components
summary: These are configurable components that can be re-used quite easily.
---

# Configurable Transformer Components
"""
import copy

import torch.nn as nn
from labml.configs import BaseConfigs, option, calculate
from labml_helpers.module import Module

from .mha import MultiHeadAttention
from .models import EmbeddingsWithPositionalEncoding, EmbeddingsWithLearnedPositionalEncoding, FeedForward, \
    TransformerLayer, Encoder, Decoder, Generator, EncoderDecoder


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
    # Number of features in position-wise feedforward layer
    d_ff: int = 2048
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
    # Position-wise feedforward layer
    feed_forward: FeedForward
    # Activation in position-wise feedforward layer
    feed_forward_activation: nn.Module = 'ReLU'

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


@option(TransformerConfigs.feed_forward_activation, 'ReLU')
def _feed_forward_activation_relu():
    """
    ReLU activation
    """
    return nn.ReLU()


@option(TransformerConfigs.feed_forward_activation, 'GELU')
def _feed_forward_activation_relu():
    """
    GELU activation
    """
    return nn.GELU()


@option(TransformerConfigs.feed_forward, 'default')
def _feed_forward(c: TransformerConfigs):
    """
    Create feedforward layer
    """
    return FeedForward(c.d_model, c.d_ff, c.dropout, c.feed_forward_activation)


# ### Multi-head Attention
def _mha(c: TransformerConfigs):
    return MultiHeadAttention(c.n_heads, c.d_model)

calculate(TransformerConfigs.encoder_attn, 'mha', _mha)
calculate(TransformerConfigs.decoder_attn, 'mha', _mha)
calculate(TransformerConfigs.decoder_mem_attn, 'mha', _mha)


# ### Relative Multi-head Attention
def _relative_mha(c: TransformerConfigs):
    from .relative_mha import RelativeMultiHeadAttention
    return RelativeMultiHeadAttention(c.n_heads, c.d_model)


calculate(TransformerConfigs.encoder_attn, 'relative', _relative_mha)
calculate(TransformerConfigs.decoder_attn, 'relative', _relative_mha)
calculate(TransformerConfigs.decoder_mem_attn, 'relative', _relative_mha)


@option(TransformerConfigs.encoder_layer, 'default')
def _encoder_layer(c: TransformerConfigs):
    """
    Encoder layer
    """
    return TransformerLayer(d_model=c.d_model, self_attn=c.encoder_attn,
                            src_attn=None, feed_forward=copy.deepcopy(c.feed_forward),
                            dropout_prob=c.dropout)


@option(TransformerConfigs.decoder_layer, 'default')
def _decoder_layer(c: TransformerConfigs):
    """
    Decoder layer
    """
    return TransformerLayer(d_model=c.d_model, self_attn=c.decoder_attn,
                            src_attn=c.decoder_mem_attn, feed_forward=copy.deepcopy(c.feed_forward),
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


# ## Positional Embeddings
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


# ## Learned Positional Embeddings
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


# ## No Positional Embeddings
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
