"""
---
title:  Hierarchical Transformers Are More Efficient Language Models Experiment
summary: This experiment trains a hourglass model on Tiny Shakespeare dataset.
---

# [Hierarchical Transformers Are More Efficient Language Models](index.html) Experiment

This is an annotated PyTorch experiment to train a [hourglass](index.html).

This is based on
[training loop and configurations for a simple transformer auto-regressive NLP task](../basic/autoregressive_experiment.html).

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/855b82363e4911ec9ae4a5b9c69d5061)
"""
import math
from typing import List

import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.transformers.hour_glass import HourGlass
from labml_nn.transformers.positional_encoding import PositionalEncoding


class AutoregressiveTransformer(Module):
    """
    ## Autoregressive language model
    """

    def __init__(self, n_tokens: int, d_model: int, dropout: float, hour_glass: HourGlass):
        """
        * `n_tokens` is the vocabulary size
        * `d_model` is the size of the token embeddings
        * `dropout` is the dropout probability
        * `hour_glass` is the [hourglass model](index.html)
        """
        super().__init__()
        # Token embeddings
        self.embedding = nn.Embedding(n_tokens, d_model)
        # [Fixed positional embeddings](../positional_encoding.html).
        #
        # 📝 The
        # [official paper implementation](https://github.com/google/trax/blob/master/trax/models/research/hourglass.py)
        # use [relative attention](../xl/relative_mha.html)
        self.pos_embedding = PositionalEncoding(d_model, dropout)
        # [hourglass model](index.html)
        self.hour_glass = hour_glass
        # To normalize the final embeddings
        self.norm = nn.LayerNorm([d_model])
        # Embedding size
        self.d_model = d_model
        # Final linear layer to predict the logits
        self.output = nn.Linear(d_model, n_tokens)

    def __call__(self, x: torch.Tensor):
        """
        * `x` is the tensor with token indexes of shape `[seq_len, batch_size]`
        """
        # Get embeddings
        x = self.embedding(x)

        # Add [positional embeddings](../positional_encoding.html)
        if self.pos_embedding is not None:
            x = self.pos_embedding(x * math.sqrt(self.d_model))

        # Hourglass
        x = self.hour_glass(x)

        # Get logits
        output = self.output(self.norm(x))

        # Return the logits
        return output, None


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    This inherits from
    [training loop and configurations for a simple transformer auto-regressive NLP task](../basic/autoregressive_transformer.html).
    """
    # Model
    model: AutoregressiveTransformer
    # Number of attention heads
    n_heads: int = 8
    # Dropout probability
    dropout: float = 0.1
    # Size of feed-forward hidden layer
    d_ff: int = 512
    # Token embedding size
    d_model: int = 256
    # Shortening factors
    shortening_factors: List[int] = [8, 4]


@option(Configs.model)
def _model(c: Configs):
    """
    Create the model
    """

    # Create hourglass model
    hour_glass = HourGlass(c.n_heads, c.d_model, c.dropout, c.d_ff, c.shortening_factors)
    # Create the auto-regressive wrapper
    m = AutoregressiveTransformer(c.n_tokens, c.d_model, c.dropout, hour_glass).to(c.device)

    #
    return m


def main():
    # Create experiment
    experiment.create(name="hour_glass")
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
        # Train for $128$ epochs
        'epochs': 128,
        # Batch size $32$
        'batch_size': 32,
        # Switch between training and validation for $10$ times
        # per epoch
        'inner_iterations': 10,

        # Use [Noam optimizer](../../optimizers/noam.html)
        'optimizer.optimizer': 'Noam',
        'optimizer.learning_rate': 1.,
        #
    })

    # Set models for saving and loading
    experiment.add_pytorch_models({'model': conf.model})

    # Start the experiment
    with experiment.start():
        # Run training
        conf.run()


#
if __name__ == '__main__':
    main()
