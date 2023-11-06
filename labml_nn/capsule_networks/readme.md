# [Capsule Networks](https://nn.labml.ai/capsule_networks/index.html)

This is a [PyTorch](https://pytorch.org) implementation/tutorial of
[Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829).

Capsule network is a neural network architecture that embeds features
as capsules and routes them with a voting mechanism to next layer of capsules.

Unlike in other implementations of models, we've included a sample, because
it is difficult to understand some concepts with just the modules.
[This is the annotated code for a model that uses capsules to classify MNIST dataset](mnist.html)

This file holds the implementations of the core modules of Capsule Networks.

I used [jindongwang/Pytorch-CapsuleNet](https://github.com/jindongwang/Pytorch-CapsuleNet) to clarify some
confusions I had with the paper.

Here's a notebook for training a Capsule Network on MNIST dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/capsule_networks/mnist.ipynb)
