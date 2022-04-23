"""
---
title: DeepNorm Experiment
summary: >
 Training a DeepNorm transformer on Tiny Shakespeare.
---

# DeepNorm Experiment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/normalization/deep_norm/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/ec8e4dacb7f311ec8d1cd37d50b05c3d)
"""

import copy

import torch
import torch.nn as nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.normalization.deep_norm import DeepNormTransformerLayer
from labml_nn.transformers import MultiHeadAttention
from labml_nn.transformers.feed_forward import FeedForward


class AutoregressiveTransformer(Module):
    """
    ## Auto-Regressive model

    This is a autoregressive transformer model that uses DeepNorm.
    """

    def __init__(self, n_tokens: int, d_model: int, n_layers: int, layer: DeepNormTransformerLayer):
        """
        :param n_tokens: is the number of tokens in the vocabulary
        :param d_model: is the embedding size
        :param n_layers: is the number of transformer layers
        :param layer: is the layer. We use `n_layers` copies of this for the tranformer.
        """
        super().__init__()
        # Transformer with `n_layers` layers
        self.transformer = nn.Sequential(*[copy.deepcopy(layer) for _ in range(n_layers)])

        # Token embedding layer
        self.emb = nn.Embedding(n_tokens, d_model)
        # Readout layer
        self.readout = nn.Linear(d_model, n_tokens)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the input tokens of shape `[seq_len, batch_size]`
        """
        # Get the token embeddings
        x = self.emb(x)
        # Transformer encoder
        x = self.transformer(x)
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
    n_layers: int = 64

    # $\alpha$ and $\beta$ for DeepNorm
    deep_norm_alpha: float
    deep_norm_beta: float

    # Number of heads in the attention
    n_heads: int = 4
    # Embedding size
    d_model: int = 64
    # Size of each attention head
    d_k: int = 16


@option(Configs.deep_norm_alpha)
def _deep_norm_alpha(c: Configs):
    """
    #### Calculate $\alpha$

    $\alpha = (2M)^{\frac{1}{4}}$
    """
    return (2. * c.n_layers) ** (1. / 4.)


@option(Configs.deep_norm_beta)
def _deep_norm_beta(c: Configs):
    """
    #### Calculate $\beta$

    $\beta = (8M)^{-\frac{1}{4}}$
    """
    return (8. * c.n_layers) ** -(1. / 4.)


@option(Configs.model)
def _model(c: Configs):
    """
    #### Initialize the model
    """
    m = AutoregressiveTransformer(c.n_tokens, c.d_model, c.n_layers,
                                  DeepNormTransformerLayer(d_model=c.d_model,
                                                           deep_norm_alpha=c.deep_norm_alpha,
                                                           deep_norm_beta=c.deep_norm_beta,
                                                           feed_forward=FeedForward(d_model=c.d_model,
                                                                                    d_ff=c.d_model * 4),
                                                           self_attn=MultiHeadAttention(c.n_heads, c.d_model,
                                                                                        dropout_prob=0.0)))

    return m.to(c.device)


def main():
    """
    #### Create and run the experiment
    """
    # Create experiment
    experiment.create(name="deep_norm", writers={'screen', 'web_api', 'comet'})
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
