# [Transformer XL](https://nn.labml.ai/transformers/xl/index.html)

This is an implementation of
[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://papers.labml.ai/paper/1901.02860)
in [PyTorch](https://pytorch.org).

Transformer has a limited attention span,
equal to the length of the sequence trained in parallel.
All these positions have a fixed positional encoding.
Transformer XL increases this attention span by letting
each of the positions pay attention to precalculated past embeddings.
For instance if the context length is $l$, it will keep the embeddings of
all layers for previous batch of length $l$ and feed them to current step.
If we use fixed-positional encodings these pre-calculated embeddings will have
the same positions as the current context.
They introduce relative positional encoding, where the positional encodings
are introduced at the attention calculation.

Annotated implementation of relative multi-headed attention is in [`relative_mha.py`](https://nn.labml.ai/transformers/xl/relative_mha.html).

Here's [the training code](https://nn.labml.ai/transformers/xl/experiment.html) and a notebook for training a transformer XL model on Tiny Shakespeare dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/xl/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/d3b6760c692e11ebb6a70242ac1c0002)
