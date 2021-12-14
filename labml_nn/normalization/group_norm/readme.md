# [Group Normalization](https://nn.labml.ai/normalization/group_norm/index.html)

This is a [PyTorch](https://pytorch.org) implementation of
the [Group Normalization](https://papers.labml.ai/paper/1803.08494) paper.

[Batch Normalization](https://nn.labml.ai/normalization/batch_norm/index.html) works well for large enough batch sizes
but not well for small batch sizes, because it normalizes over the batch.
Training large models with large batch sizes is not possible due to the memory capacity of the
devices.

This paper introduces Group Normalization, which normalizes a set of features together as a group.
This is based on the observation that classical features such as
[SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) and
[HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) are group-wise features.
The paper proposes dividing feature channels into groups and then separately normalizing
all channels within each group.

Here's a [CIFAR 10 classification model](https://nn.labml.ai/normalization/group_norm/experiment.html) that uses instance normalization.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/normalization/group_norm/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/081d950aa4e011eb8f9f0242ac1c0002)
