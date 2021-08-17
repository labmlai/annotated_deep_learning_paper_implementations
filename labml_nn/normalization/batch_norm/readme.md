# [Batch Normalization](https://nn.labml.ai/normalization/batch_norm/index.html)

This is a [PyTorch](https://pytorch.org) implementation of Batch Normalization from paper
 [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://papers.labml.ai/paper/1502.03167).

### Internal Covariate Shift

The paper defines *Internal Covariate Shift* as the change in the
distribution of network activations due to the change in
network parameters during training.
For example, let's say there are two layers $l_1$ and $l_2$.
During the beginning of the training $l_1$ outputs (inputs to $l_2$)
could be in distribution $\mathcal{N}(0.5, 1)$.
Then, after some training steps, it could move to $\mathcal{N}(0.6, 1.5)$.
This is *internal covariate shift*.

Internal covariate shift will adversely affect training speed because the later layers
($l_2$ in the above example) have to adapt to this shifted distribution.

By stabilizing the distribution, batch normalization minimizes the internal covariate shift.

## Normalization

It is known that whitening improves training speed and convergence.
*Whitening* is linearly transforming inputs to have zero mean, unit variance,
and be uncorrelated.

### Normalizing outside gradient computation doesn't work

Normalizing outside the gradient computation using pre-computed (detached)
means and variances doesn't work. For instance. (ignoring variance), let
$$\hat{x} = x - \mathbb{E}[x]$$
where $x = u + b$ and $b$ is a trained bias
and $\mathbb{E}[x]$ is an outside gradient computation (pre-computed constant).

Note that $\hat{x}$ has no effect on $b$.
Therefore,
$b$ will increase or decrease based
$\frac{\partial{\mathcal{L}}}{\partial x}$,
and keep on growing indefinitely in each training update.
The paper notes that similar explosions happen with variances.

### Batch Normalization

Whitening is computationally expensive because you need to de-correlate and
the gradients must flow through the full whitening calculation.

The paper introduces a simplified version which they call *Batch Normalization*.
First simplification is that it normalizes each feature independently to have
zero mean and unit variance:
$$\hat{x}^{(k)} = \frac{x^{(k)} - \mathbb{E}[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}$$
where $x = (x^{(1)} ... x^{(d)})$ is the $d$-dimensional input.

The second simplification is to use estimates of mean $\mathbb{E}[x^{(k)}]$
and variance $Var[x^{(k)}]$ from the mini-batch
for normalization; instead of calculating the mean and variance across the whole dataset.

Normalizing each feature to zero mean and unit variance could affect what the layer
can represent.
As an example paper illustrates that, if the inputs to a sigmoid are normalized
most of it will be within $[-1, 1]$ range where the sigmoid is linear.
To overcome this each feature is scaled and shifted by two trained parameters
$\gamma^{(k)}$ and $\beta^{(k)}$.
$$y^{(k)} =\gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$$
where $y^{(k)}$ is the output of the batch normalization layer.

Note that when applying batch normalization after a linear transform
like $Wu + b$ the bias parameter $b$ gets cancelled due to normalization.
So you can and should omit bias parameter in linear transforms right before the
batch normalization.

Batch normalization also makes the back propagation invariant to the scale of the weights
and empirically it improves generalization, so it has regularization effects too.

## Inference

We need to know $\mathbb{E}[x^{(k)}]$ and $Var[x^{(k)}]$ in order to
perform the normalization.
So during inference, you either need to go through the whole (or part of) dataset
and find the mean and variance, or you can use an estimate calculated during training.
The usual practice is to calculate an exponential moving average of
mean and variance during the training phase and use that for inference.

Here's [the training code](mnist.html) and a notebook for training
a CNN classifier that uses batch normalization for MNIST dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/normalization/batch_norm/mnist.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/011254fe647011ebbb8e0242ac1c0002)
