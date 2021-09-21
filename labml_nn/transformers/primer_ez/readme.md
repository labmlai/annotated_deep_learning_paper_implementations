# [Primer: Searching for Efficient Transformers for Language Modeling](https://nn.labml.ai/transformers/primer_ez/index.html)

This is a [PyTorch](https://pytorch.org) implementation of the paper
[Primer: Searching for Efficient Transformers for Language Modeling](https://papers.labml.ai/paper/2109.08668).

The authors do an evolutionary search for transformer architectures.
They name the architecture found using the search as Primer (PRIMitives searched transformER).
**Primer EZ** is the architecture with the two most robust modifications in Primer compared to
 the original transformer.
Primer EZ trains a lot faster than the vanilla transformer.

### Squared ReLU

The most effective modification found by the search is using a square ReLU instead of ReLU in
the [position-wise feedforward module](https://nn.labml.ai/transformers/feed_forward.html).

### Multi-DConv-Head Attention (MDHA)

The next effective modification is a depth-wise 3 X 1 convolution after multi-head projection
 for queries, keys, and values.
The convolution is along the sequence dimension and per channel (depth-wise).
To be clear, if the number of channels in each head is d_k the convolution will have 1 X 3
kernels for each of the d_k channels.

[Here is the experiment code](https://nn.labml.ai/transformers/primer_ez/experiment.html), for Primer EZ.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/30adb7aa1ab211eca7310f80a114e8a4)
