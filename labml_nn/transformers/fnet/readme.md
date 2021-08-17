# [FNet: Mixing Tokens with Fourier Transforms](https://nn.labml.ai/transformers/fnet/index.html)

This is a [PyTorch](https://pytorch.org) implementation of the paper
[FNet: Mixing Tokens with Fourier Transforms](https://papers.labml.ai/paper/2105.03824).

This paper replaces the [self-attention layer](https://nn.labml.ai/transformers//mha.html) with two
[Fourier transforms](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) to
*mix* tokens.
This is a 7X more efficient than self-attention.
The accuracy loss of using this over self-attention is about 92% for
[BERT](https://paperswithcode.com/method/bert) on
[GLUE benchmark](https://paperswithcode.com/dataset/glue).
