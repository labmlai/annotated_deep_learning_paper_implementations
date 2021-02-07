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
* [Transformer Encoder and Decoder Models](models.html)
* [Fixed positional encoding](positional_encoding.html)

## [Transformer XL](xl/index.html)
This implements Transformer XL model using
[relative multi-head attention](xl/relative_mha.html)

## [GPT Architecture](gpt)

This is an implementation of GPT-2 architecture.

## [GLU Variants](glu_variants/simple.html)

This is an implementation of the paper
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).


## [kNN-LM](knn)

This is an implementation of the paper
[Generalization through Memorization: Nearest Neighbor Language Models](https://arxiv.org/abs/1911.00172).

## [Feedback Transformer](feedback)

This is an implementation of the paper
[Accessing Higher-level Representations in Sequential Transformers with Feedback Memory](https://arxiv.org/abs/2002.09402).

## [Switch Transformer](switch)

This is a miniature implementation of the paper
[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961).
Our implementation only has a few million parameters and doesn't do model parallel distributed training.
It does single GPU training but we implement the concept of switching as described in the paper.
"""

from .configs import TransformerConfigs
from .models import TransformerLayer, Encoder, Decoder, Generator, EncoderDecoder
from .mha import MultiHeadAttention
from labml_nn.transformers.xl.relative_mha import RelativeMultiHeadAttention
