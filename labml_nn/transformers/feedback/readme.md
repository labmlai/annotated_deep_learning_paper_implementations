# [Feedback Transformer](https://nn.labml.ai/transformers/feedback/index.html)

This is a [PyTorch](https://pytorch.org) implementation of the paper
[Accessing Higher-level Representations in Sequential Transformers with Feedback Memory](https://papers.labml.ai/paper/2002.09402).

Normal transformers process tokens in parallel. Each transformer layer pays attention
to the outputs of the previous layer.
Feedback transformer pays attention to the output of all layers in previous steps.
So this adds recurrence, and we need to process token-by-token.
This slows down the training significantly (about 5X - 10X depending on the sequence length).
However, when predicting Feedback Transformer is faster because you can predict the next token
if you cache the memory vectors.

In order to speed up the training the paper discusses starting with a short sequence length and
gradually increasing it.
They also discuss using a pretrained parallel transformer as the starting point.

The original feedback transformer doesn't keep the outputs of all layers.
Instead it keeps weighted sum of the output of all layers.
This reduces the memory used for caching during prediction.
The first half of this file implements this.

The updated feedback transformer shares weights used
to calculate keys and values among the layers.
We then calculate the keys and values for each step only once and keep
them cached.
The [second half](#shared_kv) of this file implements this.
We implemented a custom PyTorch function to improve performance.

Here's [the training code](experiment.html) and a notebook for training a feedback transformer on Tiny Shakespeare dataset.

[Colab Notebook](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/feedback/experiment.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/feedback/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/d8eb9416530a11eb8fb50242ac1c0002)
