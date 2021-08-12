# [PonderNet: Learning to Ponder](https://nn.labml.ai/adaptive_computation/ponder_net/index.html)

This is a [PyTorch](https://pytorch.org) implementation of the paper
[PonderNet: Learning to Ponder](https://papers.labml.ai/paper/2107.05407).

PonderNet adapts the computation based on the input.
It changes the number of steps to take on a recurrent network based on the input.
PonderNet learns this with end-to-end gradient descent.
