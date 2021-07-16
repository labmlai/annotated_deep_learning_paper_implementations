# [Deep Residual Learning for Image Recognition (ResNet)](https://nn.labml.ai/resnet/index.html)

This is a [PyTorch](https://pytorch.org) implementation of the paper
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).

ResNets train layers as residual functions to overcome the
*degradation problem*.
The degradation problem is the accuracy of deep neural networks degrading when
the number of layers becomes very high.
The accuracy increases as the number of layers increase, then saturates,
and then starts to degrade.