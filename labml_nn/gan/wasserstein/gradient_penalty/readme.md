# [Gradient Penalty for Wasserstein GAN (WGAN-GP)](https://nn.labml.ai/gan/wasserstein/gradient_penalty/index.html)

This is an implementation of
[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028).

[WGAN](https://nn.labml.ai/gan/wasserstein/index.html) suggests
clipping weights to enforce Lipschitz constraint
on the discriminator network (critic).
This and other weight constraints like L2 norm clipping, weight normalization,
L1, L2 weight decay have problems:

1. Limiting the capacity of the discriminator
2. Exploding and vanishing gradients (without [Batch Normalization](https://nn.labml.ai/normalization/batch_norm/index.html)).

The paper [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
proposal a better way to improve Lipschitz constraint, a gradient penalty.
