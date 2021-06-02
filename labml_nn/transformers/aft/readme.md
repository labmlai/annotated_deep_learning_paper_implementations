# [An Attention Free Transformer](https://nn.labml.ai/transformers/aft/index.html)

This is a [PyTorch](https://pytorch.org) implementation of the paper
[An Attention Free Transformer](https://papers.labml.ai/paper/2105.14103).

This paper replaces the [self-attention layer](https://nn.labml.ai/transformers/mha.html) 
with a new efficient operation,
that has memory complexity of O(Td), where T is the sequence length
and $d$ is the dimensionality of embeddings.

The paper introduces AFT along with AFT-local and AFT-conv.
Here we have implemented AFT-local which pays attention to closeby tokens
in an autoregressive model.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/6348e504c3a511eba9529daa283fb495)
