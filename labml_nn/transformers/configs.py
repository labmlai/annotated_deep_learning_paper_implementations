import copy

import torch.nn as nn

from labml.configs import BaseConfigs, option, calculate
from labml_helpers.module import Module
from .mha import MultiHeadAttention
from .models import EmbeddingsWithPositionalEncoding, EmbeddingsWithLearnedPositionalEncoding, FeedForward, \
    TransformerLayer, Encoder, Decoder, Generator, EncoderDecoder


class TransformerConfigs(BaseConfigs):
    n_heads: int = 8
    d_model: int = 512
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    n_src_vocab: int
    n_tgt_vocab: int

    encoder_attn: MultiHeadAttention = 'mha'
    decoder_attn: MultiHeadAttention = 'mha'
    decoder_mem_attn: MultiHeadAttention = 'mha'
    feed_forward: FeedForward

    encoder_layer: TransformerLayer = 'normal'
    decoder_layer: TransformerLayer = 'normal'

    encoder: Encoder = 'normal'
    decoder: Decoder = 'normal'

    src_embed: Module = 'fixed_pos'
    tgt_embed: Module = 'fixed_pos'

    generator: Generator = 'default'

    encoder_decoder: EncoderDecoder


@option(TransformerConfigs.feed_forward, 'default')
def _feed_forward(c: TransformerConfigs):
    return FeedForward(c.d_model, c.d_ff, c.dropout)


### MHA
def _mha(c: TransformerConfigs):
    return MultiHeadAttention(c.n_heads, c.d_model)


calculate(TransformerConfigs.encoder_attn, 'mha', _mha)
calculate(TransformerConfigs.decoder_attn, 'mha', _mha)
calculate(TransformerConfigs.decoder_mem_attn, 'mha', _mha)


### Relative MHA
def _relative_mha(c: TransformerConfigs):
    from .relative_mha import RelativeMultiHeadAttention
    return RelativeMultiHeadAttention(c.n_heads, c.d_model)


calculate(TransformerConfigs.encoder_attn, 'relative', _relative_mha)
calculate(TransformerConfigs.decoder_attn, 'relative', _relative_mha)
calculate(TransformerConfigs.decoder_mem_attn, 'relative', _relative_mha)


@option(TransformerConfigs.encoder_layer, 'normal')
def _encoder_layer(c: TransformerConfigs):
    return TransformerLayer(d_model=c.d_model, self_attn=c.encoder_attn,
                            src_attn=None, feed_forward=copy.deepcopy(c.feed_forward),
                            dropout_prob=c.dropout)


@option(TransformerConfigs.decoder_layer, 'normal')
def _decoder_layer(c: TransformerConfigs):
    return TransformerLayer(d_model=c.d_model, self_attn=c.decoder_attn,
                            src_attn=c.decoder_mem_attn, feed_forward=copy.deepcopy(c.feed_forward),
                            dropout_prob=c.dropout)


@option(TransformerConfigs.encoder, 'normal')
def _encoder(c: TransformerConfigs):
    return Encoder(c.encoder_layer, c.n_layers)


@option(TransformerConfigs.decoder, 'normal')
def _decoder(c: TransformerConfigs):
    return Decoder(c.decoder_layer, c.n_layers)


@option(TransformerConfigs.generator, 'default')
def _generator(c: TransformerConfigs):
    return Generator(c.n_tgt_vocab, c.d_model)


### Positional Embeddings
@option(TransformerConfigs.src_embed, 'fixed_pos')
def _src_embed_with_positional(c: TransformerConfigs):
    return EmbeddingsWithPositionalEncoding(c.d_model, c.n_src_vocab)


@option(TransformerConfigs.tgt_embed, 'fixed_pos')
def _tgt_embed_with_positional(c: TransformerConfigs):
    return EmbeddingsWithPositionalEncoding(c.d_model, c.n_tgt_vocab)


### Learned Positional Embeddings
@option(TransformerConfigs.src_embed, 'learned_pos')
def _src_embed_with_learned_positional(c: TransformerConfigs):
    return EmbeddingsWithLearnedPositionalEncoding(c.d_model, c.n_src_vocab)


@option(TransformerConfigs.tgt_embed, 'learned_pos')
def _tgt_embed_with_learned_positional(c: TransformerConfigs):
    return EmbeddingsWithLearnedPositionalEncoding(c.d_model, c.n_tgt_vocab)


### No Positional Embeddings
@option(TransformerConfigs.src_embed, 'no_pos')
def _src_embed_without_positional(c: TransformerConfigs):
    return nn.Embedding(c.n_src_vocab, c.d_model)


@option(TransformerConfigs.tgt_embed, 'no_pos')
def _tgt_embed_without_positional(c: TransformerConfigs):
    return nn.Embedding(c.n_tgt_vocab, c.d_model)


@option(TransformerConfigs.encoder_decoder, 'normal')
def _encoder_decoder(c: TransformerConfigs):
    return EncoderDecoder(c.encoder, c.decoder, c.src_embed, c.tgt_embed, c.generator)
