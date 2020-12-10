"""
---
title: Transformers
summary: >
  This is a collection of PyTorch implementations/tutorials of
  transformers and related techniques.
---

# Transformers

## Transformer Building Blocks

* [Multi-head attention](mha.html)
* [Relative multi-head attention](relative_mha.html)
* [Transformer Encoder and Decoder Models](models.html)
* [Fixed positional encoding](positional_encoding.html)

## [kNN-LM](knn)

This is an implementation of the paper
 [Generalization through Memorization: Nearest Neighbor Language Models](https://arxiv.org/abs/1911.00172).
"""

from .configs import TransformerConfigs
from .models import TransformerLayer, Encoder, Decoder, Generator, EncoderDecoder
from .mha import MultiHeadAttention
from .relative_mha import RelativeMultiHeadAttention
