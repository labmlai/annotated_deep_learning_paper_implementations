# [Layer Normalization](https://nn.labml.ai/normalization/layer_norm/index.html)

This is a [PyTorch](https://pytorch.org) implementation of
[Layer Normalization](https://papers.labml.ai/paper/1607.06450).

### Limitations of [Batch Normalization](https://nn.labml.ai/normalization/batch_norm/index.html)

* You need to maintain running means.
* Tricky for RNNs. Do you need different normalizations for each step?
* Doesn't work with small batch sizes;
large NLP models are usually trained with small batch sizes.
* Need to compute means and variances across devices in distributed training.

## Layer Normalization

Layer normalization is a simpler normalization method that works
on a wider range of settings.
Layer normalization transforms the inputs to have zero mean and unit variance
across the features.
*Note that batch normalization fixes the zero mean and unit variance for each element.*
Layer normalization does it for each batch across all elements.

Layer normalization is generally used for NLP tasks.

We have used layer normalization in most of the
[transformer implementations](https://nn.labml.ai/transformers/gpt/index.html).