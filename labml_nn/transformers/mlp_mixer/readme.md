# [MLP-Mixer: An all-MLP Architecture for Vision](https://nn.labml.ai/transformers/mlp_mixer/index.html)

This is a [PyTorch](https://pytorch.org) implementation of the paper
[MLP-Mixer: An all-MLP Architecture for Vision](https://papers.labml.ai/paper/2105.01601).

This paper applies the model on vision tasks.
The model is similar to a transformer with attention layer being replaced by a MLP
that is applied across the patches (or tokens in case of a NLP task).

Our implementation of MLP Mixer is a drop in replacement for the [self-attention layer](https://nn.labml.ai/transformers/mha.html)
in [our transformer implementation](https://nn.labml.ai/transformers/models.html).
So it's just a couple of lines of code, transposing the tensor to apply the MLP
across the sequence dimension.

Although the paper applied MLP Mixer on vision tasks,
we tried it on a [masked language model](https://nn.labml.ai/transformers/mlm/index.html).
[Here is the experiment code](https://nn.labml.ai/transformers/mlp_mixer/experiment.html).

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/994263d2cdb511eb961e872301f0dbab)
