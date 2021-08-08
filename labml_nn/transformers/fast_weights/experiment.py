"""
---
title: Train Fast Weights Transformer
summary: This is training code with notes for a Fast Weights Transformer.
---

# Train Fast Weights Transformer

This trains a fast weights transformer model for auto-regression.

Here’s a Colab notebook for training a fast weights transformer on Tiny Shakespeare dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/fast_weights/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/928aadc0846c11eb85710242ac1c0002)
"""

import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml.utils.pytorch import get_modules
from labml_helpers.module import Module
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs


class AutoregressiveModel(Module):
    """
    ## Auto regressive model
    """

    def __init__(self, n_vocab: int, d_model: int, transformer: Module):
        super().__init__()
        # Token embedding module
        self.src_embed = nn.Embedding(n_vocab, d_model)
        self.transformer = transformer
        self.generator = nn.Linear(d_model, n_vocab)

    def forward(self, x: torch.Tensor):
        # Embed the tokens
        x = self.src_embed(x)
        # Run it through the the transformer
        res = self.transformer(x)
        # Generate logits of the next token
        return self.generator(res), None


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    The default configs can and will be over-ridden when we start the experiment
    """

    model: AutoregressiveModel

    d_model: int = 512
    nu: int = 1
    heads: int = 8
    dropout: float = 0.0
    d_ff: int = 2048
    n_layers: int = 6


@option(Configs.model)
def fast_weights_transformer(c: Configs):
    """
    Create [fast weights transformer](index.html).
    """
    from labml_nn.transformers.fast_weights import FastWeightsAttentionTransformer, \
        FastWeightsAttentionTransformerLayer, FastWeightsAttention, FeedForward

    from labml_nn.transformers.fast_weights import DPFP
    return AutoregressiveModel(
        c.n_tokens, c.d_model,
        FastWeightsAttentionTransformer(
            FastWeightsAttentionTransformerLayer(d_model=c.d_model,
                                                 attn=FastWeightsAttention(c.heads, c.d_model, c.dropout, DPFP(nu=c.nu)),
                                                 feed_forward=FeedForward(c.d_model, c.d_ff, c.dropout),
                                                 dropout_prob=c.dropout),
            c.n_layers)).to(c.device)


def main():
    # Create experiment
    experiment.create(name="fast_weights_transformer")
    # Create configs
    conf = Configs()
    # Load configurations
    experiment.configs(conf,
                       # A dictionary of configurations to override
                       {'tokenizer': 'character',
                        'text': 'tiny_shakespeare',
                        'optimizer.learning_rate': 1.0,
                        'optimizer.optimizer': 'Noam',
                        'prompt': 'It is',
                        'prompt_separator': '',

                        'train_loader': 'shuffled_train_loader',
                        'valid_loader': 'shuffled_valid_loader',

                        'seq_len': 128,
                        'epochs': 128,
                        'batch_size': 16,
                        'inner_iterations': 25})

    # Set models for saving and loading
    experiment.add_pytorch_models(get_modules(conf))

    # Start the experiment
    with experiment.start():
        # Run the training loop
        conf.run()


if __name__ == '__main__':
    main()
