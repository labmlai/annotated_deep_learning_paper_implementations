#  [Patches Are All You Need?](https://nn.labml.ai/conv_mixer/index.html)

This is a [PyTorch](https://pytorch.org) implementation of the paper
[Patches Are All You Need?](https://papers.labml.ai/paper/2201.09792).

ConvMixer is Similar to [MLP-Mixer](https://nn.labml.ai/transformers/mlp_mixer/index.html).
MLP-Mixer separates mixing of spatial and channel dimensions, by applying an MLP across spatial dimension
and then an MLP across the channel dimension
(spatial MLP replaces the [ViT](https://nn.labml.ai/transformers/vit/index.html) attention
and channel MLP is the [FFN](https://nn.labml.ai/transformers/feed_forward.html) of ViT).

ConvMixer uses a 1x1 convolution for channel mixing and a
depth-wise convolution for spatial mixing.
Since it's a convolution instead of a full MLP across the space, it mixes only the nearby batches in
contrast to ViT or MLP-Mixer.
Also, the MLP-mixer uses MLPs of two layers for each mixing and ConvMixer uses a single layer for each mixing.

The paper recommends removing the residual connection across the channel mixing (point-wise convolution)
and having only a residual connection over the spatial mixing (depth-wise convolution).
They also use [Batch normalization](https://nn.labml.ai/normalization/batch_norm/index.html) instead
of [Layer normalization](../normalization/layer_norm/index.html).

Here's [an experiment](https://nn.labml.ai/conv_mixer/experiment.html) that trains ConvMixer on CIFAR-10.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/0fc344da2cd011ecb0bc3fdb2e774a3d)