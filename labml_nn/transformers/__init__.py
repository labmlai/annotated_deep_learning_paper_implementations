"""
---
title: Transformers
summary: >
  This is a collection of PyTorch implementations/tutorials of
  transformers and related techniques.
---

# Transformers

This module contains [PyTorch](https://pytorch.org/)
implementations and explanations of original transformer
from paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
and derivatives and enhancements of it.

* [Multi-head attention](mha.html)
* [Relative multi-head attention](relative_mha.html)
* [Transformer Encoder and Decoder Models](models.html)
* [Fixed positional encoding](positional_encoding.html)

## [kNN-LM](knn)

This is an implementation of the paper
[Generalization through Memorization: Nearest Neighbor Language Models](https://arxiv.org/abs/1911.00172).

## [Feedback Transformer](feedback)

This is an implementation of the paper
[Accessing Higher-level Representations in Sequential Transformers with Feedback Memory](https://arxiv.org/abs/2002.09402).
"""

from .configs import TransformerConfigs
from .models import TransformerLayer, Encoder, Decoder, Generator, EncoderDecoder
from .mha import MultiHeadAttention
from .relative_mha import RelativeMultiHeadAttention
