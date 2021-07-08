"""
---
title: Graph Attention Networks (GAT)
summary: >
 A PyTorch implementation/tutorial of Graph Attention Networks.
---

# Graph Attention Networks (GAT)

This is a [PyTorch](https://pytorch.org) implementation of the paper
[Graph Attention Networks](https://arxiv.org/abs/1710.10903).

GATs work on graph data.
A graph consists of nodes and edges connecting nodes.
For example, in Cora dataset the nodes are research papers and the edges are citations that
connect the papers.

GAT uses masked self-attention, kind of similar to [transformers](../../transformers/mha.html).
GAT consists of graph attention layers stacked on top of each other.
Each graph attention layer gets node embeddings as inputs and outputs transformed embeddings.
The node embeddings pay attention to the embeddings of other nodes it's connected to.
The details of graph attention layers are included alongside the implementation.

Here is [the training code](experiment.html) for training
a two-layer GAT on Cora dataset.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/d6c636cadf3511eba2f1e707f612f95d)
"""

import torch
from torch import nn

from labml_helpers.module import Module


class GraphAttentionLayer(Module):
    """
    ## Graph attention layer

    This is a single graph attention layer.
    A GAT is made up of multiple such layers.

    It takes
    $$\mathbf{h} = \{ \overrightarrow{h_1}, \overrightarrow{h_2}, \dots, \overrightarrow{h_N} \}$$,
    where $\overrightarrow{h_i} \in \mathbb{R}^F$ as input
    and outputs
    $$\mathbf{h'} = \{ \overrightarrow{h'_1}, \overrightarrow{h'_2}, \dots, \overrightarrow{h'_N} \}$$,
    where $\overrightarrow{h'_i} \in \mathbb{R}^{F'}$.
    """
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        """
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial transformation;
        # i.e. to transform the node embeddings before self-attention
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def __call__(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `h`, $\mathbf{h}$ is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.

        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """

        # Number of nodes
        n_nodes = h.shape[0]
        # The initial transformation,
        # $$\overrightarrow{g^k_i} = \mathbf{W}^k \overrightarrow{h_i}$$
        # for each head.
        # We do single linear transformation and then split it up for each head.
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)

        # #### Calculate attention score
        #
        # We calculate these for each head $k$. *We have omitted $\cdot^k$ for simplicity*.
        #
        # $$e_{ij} = a(\mathbf{W} \overrightarrow{h_i}, \mathbf{W} \overrightarrow{h_j}) =
        # a(\overrightarrow{g_i}, \overrightarrow{g_j})$$
        #
        # $e_{ij}$ is the attention score (importance) from node $j$ to node $i$.
        # We calculate this for each head.
        #
        # $a$ is the attention mechanism, that calculates the attention score.
        # The paper concatenates
        # $\overrightarrow{g_i}$, $\overrightarrow{g_j}$
        # and does a linear transformation with a weight vector $\mathbf{a} \in \mathbb{R}^{2 F'}$
        # followed by a $\text{LeakyReLU}$.
        #
        # $$e_{ij} = \text{LeakyReLU} \Big(
        # \mathbf{a}^\top \Big[
        # \overrightarrow{g_i} \Vert \overrightarrow{g_j}
        # \Big] \Big)$$

        # First we calculate
        # $\Big[\overrightarrow{g_i} \Vert \overrightarrow{g_j} \Big]$
        # for all pairs of $i, j$.
        #
        # `g_repeat` gets
        # $$\{\overrightarrow{g_1}, \overrightarrow{g_2}, \dots, \overrightarrow{g_N},
        # \overrightarrow{g_1}, \overrightarrow{g_2}, \dots, \overrightarrow{g_N}, ...\}$$
        # where each node embedding is repeated `n_nodes` times.
        g_repeat = g.repeat(n_nodes, 1, 1)
        # `g_repeat_interleave` gets
        # $$\{\overrightarrow{g_1}, \overrightarrow{g_1}, \dots, \overrightarrow{g_1},
        # \overrightarrow{g_2}, \overrightarrow{g_2}, \dots, \overrightarrow{g_2}, ...\}$$
        # where each node embedding is repeated `n_nodes` times.
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        # Now we concatenate to get
        # $$\{\overrightarrow{g_1} \Vert \overrightarrow{g_1},
        # \overrightarrow{g_1}, \Vert \overrightarrow{g_2},
        # \dots, \overrightarrow{g_1}  \Vert \overrightarrow{g_N},
        # \overrightarrow{g_2} \Vert \overrightarrow{g_1},
        # \overrightarrow{g_2}, \Vert \overrightarrow{g_2},
        # \dots, \overrightarrow{g_2}  \Vert \overrightarrow{g_N}, ...\}$$
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        # Reshape so that `g_concat[i, j]` is $\overrightarrow{g_i} \Vert \overrightarrow{g_j}$
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)

        # Calculate
        # $$e_{ij} = \text{LeakyReLU} \Big(
        # \mathbf{a}^\top \Big[
        # \overrightarrow{g_i} \Vert \overrightarrow{g_j}
        # \Big] \Big)$$
        # `e` is of shape `[n_nodes, n_nodes, n_heads, 1]`
        e = self.activation(self.attn(g_concat))
        # Remove the last dimension of size `1`
        e = e.squeeze(-1)

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        # Mask $e_{ij}$ based on adjacency matrix.
        # $e_{ij}$ is set to $- \infty$ if there is no edge from $i$ to $j$.
        e = e.masked_fill(adj_mat == 0, float('-inf'))

        # We then normalize attention scores (or coefficients)
        # $$\alpha_{ij} = \text{softmax}_j(e_{ij}) =
        # \frac{\exp(e_{ij})}{\sum_{j \in \mathcal{N}_i} \exp(e_{ij})}$$
        #
        # where $\mathcal{N}_i$ is the set of nodes connected to $i$.
        #
        # We do this by setting unconnected $e_{ij}$ to $- \infty$ which
        # makes $\exp(e_{ij}) \sim 0$ for unconnected pairs.
        a = self.softmax(e)

        # Apply dropout regularization
        a = self.dropout(a)

        # Calculate final output for each head
        # $$\overrightarrow{h'^k_i} = \sum_{j \in \mathcal{N}_i} \alpha^k_{ij} \overrightarrow{g^k_j}$$
        #
        # *Note:* The paper includes the final activation $\sigma$ in $\overrightarrow{h_i}$
        # We have omitted this from the Graph Attention Layer implementation
        # and use it on the GAT model to match with how other PyTorch modules are defined -
        # activation as a separate layer.
        attn_res = torch.einsum('ijh,jhf->ihf', a, g)

        # Concatenate the heads
        if self.is_concat:
            # $$\overrightarrow{h'_i} = \Bigg\Vert_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            # $$\overrightarrow{h'_i} = \frac{1}{K} \sum_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.mean(dim=1)
