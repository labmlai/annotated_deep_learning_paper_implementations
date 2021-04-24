# [Group Normalization](https://nn.labml.ai/normalization/group_norm/index.html)

This is a [PyTorch](https://pytorch.org) implementation of
the [Group Normalization](https://arxiv.org/abs/1803.08494) paper.

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

## Formulation

All normalization layers can be defined by the following computation.

$$\hat{x}_i = \frac{1}{\sigma_i} (x_i - \mu_i)$$

where $x$ is the tensor representing the batch,
and $i$ is the index of a single value.
For instance, when it's 2D images
$i = (i_N, i_C, i_H, i_W)$ is a 4-d vector for indexing
image within batch, feature channel, vertical coordinate and horizontal coordinate.
$\mu_i$ and $\sigma_i$ are mean and standard deviation.

\begin{align}
\mu_i &= \frac{1}{m} \sum_{k \in \mathcal{S}_i} x_k \\
\sigma_i  &= \sqrt{\frac{1}{m} \sum_{k \in \mathcal{S}_i} (x_k - \mu_i)^2 + \epsilon}
\end{align}

$\mathcal{S}_i$ is the set of indexes across which the mean and standard deviation
are calculated for index $i$.
$m$ is the size of the set $\mathcal{S}_i$ which is the same for all $i$.

The definition of $\mathcal{S}_i$ is different for
[Batch normalization](https://nn.labml.ai/normalization/batch_norm/index.html),
[Layer normalization](https://nn.labml.ai/normalization/layer_norm/index.html), and
[Instance normalization](https://nn.labml.ai/normalization/instance_norm/index.html).

### [Batch Normalization](https://nn.labml.ai/normalization/batch_norm/index.html)

$$\mathcal{S}_i = \{k | k_C = i_C\}$$

The values that share the same feature channel are normalized together.

### [Layer Normalization](https://nn.labml.ai/normalization/layer_norm/index.html)

$$\mathcal{S}_i = \{k | k_N = i_N\}$$

The values from the same sample in the batch are normalized together.

### [Instance Normalization](https://nn.labml.ai/normalization/instance_norm/index.html)

$$\mathcal{S}_i = \{k | k_N = i_N, k_C = i_C\}$$

The values from the same sample and same feature channel are normalized together.

### Group Normalization

$$\mathcal{S}_i = \{k | k_N = i_N,
 \bigg \lfloor \frac{k_C}{C/G} \bigg \rfloor = \bigg \lfloor \frac{i_C}{C/G} \bigg \rfloor\}$$

where $G$ is the number of groups and $C$ is the number of channels.

Group normalization normalizes values of the same sample and the same group of channels together.

Here's a [CIFAR 10 classification model](https://nn.labml.ai/normalization/group_norm/experiment.html) that uses instance normalization.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/nn/blob/master/labml_nn/normalization/group_norm/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/081d950aa4e011eb8f9f0242ac1c0002)
[![WandB](https://img.shields.io/badge/wandb-run-yellow)](https://wandb.ai/vpj/cifar10/runs/310etthp)