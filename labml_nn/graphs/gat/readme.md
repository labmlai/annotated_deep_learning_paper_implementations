# [Graph Attention Networks (GAT)](https://nn.labml.ai/graphs/gat/index.html)

This is a [PyTorch](https://pytorch.org) implementation of the paper
[Graph Attention Networks](https://papers.labml.ai/paper/1710.10903).

GATs work on graph data.
A graph consists of nodes and edges connecting nodes.
For example, in Cora dataset the nodes are research papers and the edges are citations that
connect the papers.

GAT uses masked self-attention, kind of similar to [transformers](https://nn.labml.ai/transformers/mha.html).
GAT consists of graph attention layers stacked on top of each other.
Each graph attention layer gets node embeddings as inputs and outputs transformed embeddings.
The node embeddings pay attention to the embeddings of other nodes it's connected to.
The details of graph attention layers are included alongside the implementation.

Here is [the training code](https://nn.labml.ai/graphs/gat/experiment.html) for training
a two-layer GAT on Cora dataset.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/d6c636cadf3511eba2f1e707f612f95d)
