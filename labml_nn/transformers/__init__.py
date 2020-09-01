import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from labml.configs import BaseConfigs, option, calculate
from labml_helpers.module import Module
from labml_nn.utils import clone_module_list
from .mha import MultiHeadAttention
from .positional_encoding import PositionalEncoding, get_positional_encoding


class EmbeddingsWithPositionalEncoding(Module):
    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len))

    def __call__(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]].requires_grad_(False)
        return self.linear(x) * math.sqrt(self.d_model) + pe


class EmbeddingsWithLearnedPositionalEncoding(Module):
    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model))

    def __call__(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]]
        return self.linear(x) * math.sqrt(self.d_model) + pe


class FeedForward(Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: torch.Tensor):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.layer2(x)


class TransformerLayer(Module):
    def __init__(self, *,
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 src_attn: MultiHeadAttention = None,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def __call__(self, *,
                 x: torch.Tensor,
                 mask: torch.Tensor,
                 src: torch.Tensor = None,
                 src_mask: torch.Tensor = None):
        z = self.norm_self_attn(x)
        attn_self = self.self_attn(query=z, key=z, value=z, mask=mask)
        x = x + self.dropout(attn_self)

        if src is not None:
            z = self.norm_src_attn(x)
            attn_src = self.src_attn(query=z, key=src, value=src, mask=src_mask)
            x = x + self.dropout(attn_src)

        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)

        # guard(x.shape, attn_self.shape, attn_src.shape, ff.shape,
        #       '_batch_size', '_seq_len', 'd_model')

        return x


class Encoder(Module):
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm([layer.size])

    def __call__(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x=x, mask=mask)
        return self.norm(x)


class Decoder(Module):
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm([layer.size])

    def __call__(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)
        return self.norm(x)


class Generator(Module):
    def __init__(self, n_vocab: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(d_model, n_vocab)

    def __call__(self, x):
        return F.log_softmax(self.projection(x), dim=-1)


class EncoderDecoder(Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: Module, tgt_embed: Module, generator: Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def __call__(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor,
                 tgt_mask: torch.Tensor):
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


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
