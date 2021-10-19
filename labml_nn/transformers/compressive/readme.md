# [Compressive Transformer](https://nn.labml.ai/transformers/compressive/index.html)

This is an implementation of
[Compressive Transformers for Long-Range Sequence Modelling](https://papers.labml.ai/paper/1911.05507)
in [PyTorch](https://pytorch.org).

This is an extension of [Transformer XL](https://nn.labml.ai/transformers/xl/index.html) where past memories
are compressed to give a longer attention range.
That is, the furthest $n_{cm} c$ memories are compressed into
$n_{cm}$ memories, where $c$ is the compression rate.

## Compression operation

The compression operation is defined as
$f_c: \mathbb{R}^{nc \times d} \rightarrow \mathbb{R}^{n \times d}$.
The paper introduces multiple choices for $f_c$ and we have only implemented
1D convolution which seems to give the best results.
Each layer has a separate compression operation $f_c^{(i)}$ where
$i$ is the layer number.

## Training compression operation

Since training compression with BPTT requires maintaining
a very large computational graph (many time steps), the paper proposes
an *auto-encoding loss* and an *attention reconstruction loss*.
The auto-encoding loss decodes the original memories from the compressed memories
and calculates the loss.
Attention reconstruction loss computes the multi-headed attention results
on the compressed memory and on uncompressed memory and gets a mean squared error
between them.
We have implemented the latter here since it gives better results.

This implementation uses pre-layer normalization
while the paper uses post-layer normalization.
Pre-layer norm does the layer norm before [FFN](../feedforward.html) and
self-attention, and the pass-through in the residual connection is not normalized.
This is supposed to be more stable in standard transformer setups.

Here are [the training code](https://nn.labml.ai/transformers/compressive/experiment.html) and a notebook for training a compressive transformer
model on the Tiny Shakespeare dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/compressive/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/0d9b5338726c11ebb7c80242ac1c0002)
