"""
---
title: Train a Graph Attention Network v2 (GATv2) on Cora dataset
summary: >
  This trains is a  Graph Attention Network v2 (GATv2) on Cora dataset
---

# Train a Graph Attention Network v2 (GATv2) on Cora dataset

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/34b1e2f6ed6f11ebb860997901a2d1e3)
"""

import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.graphs.gat.experiment import Configs as GATConfigs
from labml_nn.graphs.gatv2 import GraphAttentionV2Layer


class GATv2(Module):
    """
    ## Graph Attention Network v2 (GATv2)

    This graph attention network has two [graph attention layers](index.html).
    """

    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float,
                 share_weights: bool = True):
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_classes` is the number of classes
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        * `share_weights` if set to True, the same matrix will be applied to the source and the target node of every edge
        """
        super().__init__()

        # First graph attention layer where we concatenate the heads
        self.layer1 = GraphAttentionV2Layer(in_features, n_hidden, n_heads,
                                            is_concat=True, dropout=dropout, share_weights=share_weights)
        # Activation function after first graph attention layer
        self.activation = nn.ELU()
        # Final graph attention layer where we average the heads
        self.output = GraphAttentionV2Layer(n_hidden, n_classes, 1,
                                            is_concat=False, dropout=dropout, share_weights=share_weights)
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `x` is the features vectors of shape `[n_nodes, in_features]`
        * `adj_mat` is the adjacency matrix of the form
         `[n_nodes, n_nodes, n_heads]` or `[n_nodes, n_nodes, 1]`
        """
        # Apply dropout to the input
        x = self.dropout(x)
        # First graph attention layer
        x = self.layer1(x, adj_mat)
        # Activation function
        x = self.activation(x)
        # Dropout
        x = self.dropout(x)
        # Output layer (without activation) for logits
        return self.output(x, adj_mat)


class Configs(GATConfigs):
    """
    ## Configurations

    Since the experiment is same as [GAT experiment](../gat/experiment.html) but with
    [GATv2 model](index.html) we extend the same configs and change the model.
    """

    # Whether to share weights for source and target nodes of edges
    share_weights: bool = False
    # Set the model
    model: GATv2 = 'gat_v2_model'


@option(Configs.model)
def gat_v2_model(c: Configs):
    """
    Create GATv2 model
    """
    return GATv2(c.in_features, c.n_hidden, c.n_classes, c.n_heads, c.dropout, c.share_weights).to(c.device)


def main():
    # Create configurations
    conf = Configs()
    # Create an experiment
    experiment.create(name='gatv2')
    # Calculate configurations.
    experiment.configs(conf, {
        # Adam optimizer
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 5e-3,
        'optimizer.weight_decay': 5e-4,

        'dropout': 0.7,
    })

    # Start and watch the experiment
    with experiment.start():
        # Run the training
        conf.run()


#
if __name__ == '__main__':
    main()
