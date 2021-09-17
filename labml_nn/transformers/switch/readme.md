# [Switch Transformer](https://nn.labml.ai/transformers/switch/index.html)

This is a miniature [PyTorch](https://pytorch.org) implementation of the paper
[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://papers.labml.ai/paper/2101.03961).
Our implementation only has a few million parameters and doesn't do model parallel distributed training.
It does single GPU training, but we implement the concept of switching as described in the paper.

The Switch Transformer uses different parameters for each token by switching among parameters
based on the token.
Therefore, only a fraction of parameters are chosen for each token.
So you can have more parameters but less computational cost.

The switching happens at the Position-wise Feedforward network (FFN) of each transformer block.
Position-wise feedforward network consists of two sequentially fully connected layers.
In switch transformer we have multiple FFNs (multiple experts),
and we chose which one to use based on a router.
The output is a set of probabilities for picking a FFN,
and we pick the one with the highest probability and only evaluate that.
So essentially the computational cost is the same as having a single FFN.
In our implementation this doesn't parallelize well when you have many or large FFNs since it's all
happening on a single GPU.
In a distributed setup you would have each FFN (each very large) on a different device.

The paper introduces another loss term to balance load among the experts (FFNs) and
discusses dropping tokens when routing is not balanced.

Here's [the training code](experiment.html) and a notebook for training a switch transformer on Tiny Shakespeare dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/switch/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/353770ce177c11ecaa5fb74452424f46)
