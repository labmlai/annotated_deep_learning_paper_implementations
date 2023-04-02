# [Distilling the Knowledge in a Neural Network](https://nn.labml.ai/distillation/index.html)

This is a [PyTorch](https://pytorch.org) implementation/tutorial of the paper
[Distilling the Knowledge in a Neural Network](https://papers.labml.ai/paper/1503.02531).

It's a way of training a small network using the knowledge in a trained larger network;
i.e. distilling the knowledge from the large network.

A large model with regularization or an ensemble of models (using dropout) generalizes
better than a small model when trained directly on the data and labels.
However, a small model can be trained to generalize better with help of a large model.
Smaller models are better in production: faster, less compute, less memory.

The output probabilities of a trained model give more information than the labels
because it assigns non-zero probabilities to incorrect classes as well.
These probabilities tell us that a sample has a chance of belonging to certain classes.
For instance, when classifying digits, when given an image of digit *7*,
a generalized model will give a high probability to 7 and a small but non-zero
probability to 2, while assigning almost zero probability to other digits.
Distillation uses this information to train a small model better.
