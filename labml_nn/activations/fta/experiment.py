"""
---
title: Fuzzy Tiling Activation Experiment
summary: >
 Training a transformer with FTA in FFN on Tiny Shakespeare.
---

# [Fuzzy Tiling Activation](index.html) Experiment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/activations/fta/experiment.ipynb)

Here we train a transformer that uses [Fuzzy Tiling Activation](index.html) in the
[Feed-Forward Network](../../transformers/feed_forward.html).
We use it for a language model and train it on Tiny Shakespeare dataset
for demonstration.

However, this is probably not the ideal task for FTA, and we
believe FTA is more suitable for modeling data with continuous variables.
"""

import copy

import torch
import torch.nn as nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.activations.fta import FTA
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.transformers import MultiHeadAttention, TransformerLayer
from labml_nn.transformers.utils import subsequent_mask


class FeedForwardFTA(nn.Module):
    """
    ## FFN module with [FTA](index.html) activation
    """

    def __init__(self, d_model: int, d_ff: int,
                 activation: FTA,
                 dropout: float = 0.1):
        """
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `activation` is FTA activation module
        * `dropout` is dropout probability for the hidden layer
        """
        super().__init__()
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer1 = nn.Linear(d_model, d_ff)
        # Layer two parameterized by weight $W_1$ and bias $b_1$
        self.layer2 = nn.Linear(d_ff * activation.expansion_factor, d_model)
        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)
        # Activation function $f$
        self.activation = activation

    def forward(self, x: torch.Tensor):
        # $f(x W_1 + b_1)$
        x = self.activation(self.layer1(x))
        # Apply dropout
        x = self.dropout(x)
        #
        return self.layer2(x)


class AutoregressiveTransformer(Module):
    """
    ## Auto-Regressive model

    This is an autoregressive transformer model that uses Feed-Forward Networks with
     (Fuzzy Tiling Activations)(index.html).
    """

    def __init__(self, n_tokens: int, d_model: int, n_layers: int, layer: TransformerLayer):
        """
        :param n_tokens: is the number of tokens in the vocabulary
        :param d_model: is the embedding size
        :param n_layers: is the number of transformer layers
        :param layer: is the layer. We use `n_layers` copies of this for the transformer.
        """
        super().__init__()
        # Transformer with `n_layers` layers
        self.transformer_layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

        # Token embedding layer
        self.emb = nn.Embedding(n_tokens, d_model)
        # Readout layer
        self.readout = nn.Linear(d_model, n_tokens)

        # The mask will be initialized on the first call
        self.mask = None

    def forward(self, x: torch.Tensor):
        """
        :param x: are the input tokens of shape `[seq_len, batch_size]`
        """
        # Create auto-regressive mask
        if self.mask is None or self.mask.size(0) != len(x):
            # Subsequent mask, will mask out tokens from seeing future tokens
            self.mask = subsequent_mask(len(x)).to(x.device)

        # Get the token embeddings
        x = self.emb(x)
        # Transformer encoder
        for layer in self.transformer_layers:
            x = layer(x=x, mask=self.mask)
        # Get logits
        x = self.readout(x)

        # Return results
        return x, None


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    This inherits from
    [`NLPAutoRegressionConfigs`](../../experiments/nlp_autoregression.html#NLPAutoRegressionConfigs)
    """

    # Model
    model: AutoregressiveTransformer

    # Number of layers
    n_layers: int = 4

    # $\alpha$ and $\beta$ for DeepNorm
    deep_norm_alpha: float
    deep_norm_beta: float

    # Number of heads in the attention
    n_heads: int = 4
    # Embedding size
    d_model: int = 256
    # Size of each attention head
    d_k: int = 16
    # Feed forward layer size
    d_ff: int = 256

    # FTA
    fta_lower_limit: float = -1.
    fta_upper_limit: float = +1.
    fta_delta: float = 0.2
    fta_eta: float = 0.05


@option(Configs.model)
def _model(c: Configs):
    """
    #### Initialize the model
    """

    # Create FTA activation module
    fta = FTA(c.fta_lower_limit, c.fta_upper_limit, c.fta_delta, c.fta_eta)
    # Create the transformer.
    # We re-use [`TransformerLayer`](../../transformers/models.html#TransformerLayer) and
    # [`MultiHeadAttention`](../../transformers/mha.html) implementations.
    m = AutoregressiveTransformer(c.n_tokens, c.d_model, c.n_layers,
                                  TransformerLayer(d_model=c.d_model,
                                                   feed_forward=FeedForwardFTA(d_model=c.d_model,
                                                                               d_ff=c.d_ff,
                                                                               activation=fta,
                                                                               dropout=0.1),
                                                   self_attn=MultiHeadAttention(c.n_heads, c.d_model,
                                                                                dropout_prob=0.0),
                                                   dropout_prob=0.0))

    # Move to the device
    return m.to(c.device)


def main():
    """
    #### Create and run the experiment
    """
    # Create experiment
    experiment.create(name="fta", writers={'screen', 'labml'})
    # Create configs
    conf = Configs()
    # Override configurations
    experiment.configs(conf, {
        # Use character level tokenizer
        'tokenizer': 'character',
        # Prompt separator is blank
        'prompt_separator': '',
        # Starting prompt for sampling
        'prompt': 'It is ',
        # Use Tiny Shakespeare dataset
        'text': 'tiny_shakespeare',

        # Use a context size of $256$
        'seq_len': 256,
        # Train for 32 epochs
        'epochs': 32,
        # Batch size $16$
        'batch_size': 16,
        # Switch between training and validation for $10$ times per epoch
        'inner_iterations': 10,

        # Adam optimizer with no warmup
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 3e-4,
    })

    # Set model(s) for saving and loading
    experiment.add_pytorch_models({'model': conf.model})

    # Start the experiment
    with experiment.start():
        # Run training
        conf.run()


#
if __name__ == '__main__':
    main()
