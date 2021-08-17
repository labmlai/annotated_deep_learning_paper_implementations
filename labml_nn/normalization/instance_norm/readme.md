# [Instance Normalization](https://nn.labml.ai/normalization/instance_norm/index.html)

This is a [PyTorch](https://pytorch.org) implementation of
[Instance Normalization: The Missing Ingredient for Fast Stylization](https://papers.labml.ai/paper/1607.08022).

Instance normalization was introduced to improve [style transfer](https://paperswithcode.com/task/style-transfer).
It is based on the observation that stylization should not depend on the contrast of the content image.
Since it's hard for a convolutional network to learn "contrast normalization", this paper
introduces instance normalization which does that.