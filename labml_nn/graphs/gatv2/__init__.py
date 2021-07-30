"""
---
title: Graph Attention Networks v2 (GATv2)
summary: >
 A PyTorch implementation/tutorial of Graph Attention Networks v2.
---
# Graph Attention Networks v2 (GATv2)
This is a [PyTorch](https://pytorch.org) implementation of the GATv2 operator from the paper
[How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491).

GATv2s work on graph data similar to [GAT](../gat/index.html).
A graph consists of nodes and edges connecting nodes.
For example, in Cora dataset the nodes are research papers and the edges are citations that
connect the papers.

The GATv2 operator fixes the static attention problem of the standard [GAT](../gat/index.html).
Static attention is when the attention to the key nodes has the same rank (order) for any query node.
[GAT](../gat/index.html) computes attention from query node $i$ to key node $j$ as,

\begin{align}
e_{ij} &= \text{LeakyReLU} \Big(\mathbf{a}^\top \Big[
 \mathbf{W} \overrightarrow{h_i} \Vert  \mathbf{W} \overrightarrow{h_j}
\Big] \Big) \\
&=
\text{LeakyReLU} \Big(\mathbf{a}_1^\top  \mathbf{W} \overrightarrow{h_i} +
 \mathbf{a}_2^\top  \mathbf{W} \overrightarrow{h_j}
\Big)
\end{align}

Note that for any query node $i$, the attention rank ($argsort$) of keys depends only
on $\mathbf{a}_2^\top  \mathbf{W} \overrightarrow{h_j}$.
Therefore the attention rank of keys remains the same (*static*) for all queries.

GATv2 allows dynamic attention by changing the attention mechanism,

\begin{align}
e_{ij} &= \mathbf{a}^\top \text{LeakyReLU} \Big( \mathbf{W} \Big[
 \overrightarrow{h_i} \Vert  \overrightarrow{h_j}
\Big] \Big) \\
&= \mathbf{a}^\top \text{LeakyReLU} \Big(
\mathbf{W}_l \overrightarrow{h_i} +  \mathbf{W}_r \overrightarrow{h_j}
\Big)
\end{align}

The paper shows that GATs static attention mechanism fails on some graph problems
with a synthetic dictionary lookup dataset.
It's a fully connected bipartite graph where one set of nodes (query nodes)
have a key associated with it
and the other set of nodes have both a key and a value associated with it.
The goal is to predict the values of query nodes.
GAT fails on this task because of its limited static attention.

Here is [the training code](experiment.html) for training
a two-layer GATv2 on Cora dataset.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/34b1e2f6ed6f11ebb860997901a2d1e3)
"""

import torch
from torch import nn

from labml_helpers.module import Module


class GraphAttentionV2Layer(Module):
    """
    ## Graph attention v2 layer
    This is a single graph attention v2 layer.
    A GATv2 is made up of multiple such layers.
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
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):
        """
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        * `share_weights` if set to `True`, the same matrix will be applied to the source and the target node of every edge
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial source transformation;
        # i.e. to transform the source node embeddings before self-attention
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # If `share_weights` is `True` the same linear layer is used for the target nodes
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
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
        # The initial transformations,
        # $$\overrightarrow{{g_l}^k_i} = \mathbf{W_l}^k \overrightarrow{h_i}$$
        # $$\overrightarrow{{g_r}^k_i} = \mathbf{W_r}^k \overrightarrow{h_i}$$
        # for each head.
        # We do two linear transformations and then split it up for each head.
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)

        # #### Calculate attention score
        #
        # We calculate these for each head $k$. *We have omitted $\cdot^k$ for simplicity*.
        #
        # $$e_{ij} = a(\mathbf{W_l} \overrightarrow{h_i}, \mathbf{W_r} \overrightarrow{h_j}) =
        # a(\overrightarrow{{g_l}_i}, \overrightarrow{{g_r}_j})$$
        #
        # $e_{ij}$ is the attention score (importance) from node $j$ to node $i$.
        # We calculate this for each head.
        #
        # $a$ is the attention mechanism, that calculates the attention score.
        # The paper sums
        # $\overrightarrow{{g_l}_i}$, $\overrightarrow{{g_r}_j}$
        # followed by a $\text{LeakyReLU}$ 
        # and does a linear transformation with a weight vector $\mathbf{a} \in \mathbb{R}^{F'}$
        # 
        #
        # $$e_{ij} = \mathbf{a}^\top \text{LeakyReLU} \Big(
        # \Big[
        # \overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j}
        # \Big] \Big)$$
        # Note: The paper desrcibes $e_{ij}$ as         
        # $$e_{ij} = \mathbf{a}^\top \text{LeakyReLU} \Big( \mathbf{W}
        # \Big[
        # \overrightarrow{h_i} \Vert \overrightarrow{h_j}
        # \Big] \Big)$$
        # which is equivalent to the definition we use here.

        # First we calculate
        # $\Big[\overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j} \Big]$
        # for all pairs of $i, j$.
        #
        # `g_l_repeat` gets
        # $$\{\overrightarrow{{g_l}_1}, \overrightarrow{{g_l}_2}, \dots, \overrightarrow{{g_l}_N},
        # \overrightarrow{{g_l}_1}, \overrightarrow{{g_l}_2}, \dots, \overrightarrow{{g_l}_N}, ...\}$$
        # where each node embedding is repeated `n_nodes` times.
        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        # `g_r_repeat_interleave` gets
        # $$\{\overrightarrow{{g_r}_1}, \overrightarrow{{g_r}_1}, \dots, \overrightarrow{{g_r}_1},
        # \overrightarrow{{g_r}_2}, \overrightarrow{{g_r}_2}, \dots, \overrightarrow{{g_r}_2}, ...\}$$
        # where each node embedding is repeated `n_nodes` times.
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)
        # Now we add the two tensors to get
        # $$\{\overrightarrow{{g_l}_1} + \overrightarrow{{g_r}_1},
        # \overrightarrow{{g_l}_1}, + \overrightarrow{{g_r}_2},
        # \dots, \overrightarrow{{g_l}_1}  +\overrightarrow{{g_r}_N},
        # \overrightarrow{{g_l}_2} + \overrightarrow{{g_r}_1},
        # \overrightarrow{{g_l}_2}, + \overrightarrow{{g_r}_2},
        # \dots, \overrightarrow{{g_l}_2}  + \overrightarrow{{g_r}_N}, ...\}$$
        g_sum = g_l_repeat + g_r_repeat_interleave
        # Reshape so that `g_sum[i, j]` is $\overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j}$
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)

        # Calculate
        # $$e_{ij} = \mathbf{a}^\top \text{LeakyReLU} \Big(
        # \Big[
        # \overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j}
        # \Big] \Big)$$
        # `e` is of shape `[n_nodes, n_nodes, n_heads, 1]`
        e = self.attn(self.activation(g_sum))
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
        # $$\overrightarrow{h'^k_i} = \sum_{j \in \mathcal{N}_i} \alpha^k_{ij} \overrightarrow{{g_r}_{j,k}}$$
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)

        # Concatenate the heads
        if self.is_concat:
            # $$\overrightarrow{h'_i} = \Bigg\Vert_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            # $$\overrightarrow{h'_i} = \frac{1}{K} \sum_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.mean(dim=1)
