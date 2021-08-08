"""
---
title: Train Feedback Transformer
summary: This is training code with notes for a feedback transformer.
---

# Train Feedback Transformer

This trains a [feedback transformer](index.html) model for auto-regression.
You can pick the original feedback transformer or the new version
where the keys and values are precalculated.

Here's a Colab notebook for training a feedback transformer on Tiny Shakespeare dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/feedback/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/d8eb9416530a11eb8fb50242ac1c0002)
"""

import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml.utils.pytorch import get_modules
from labml_helpers.module import Module

from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.transformers import Encoder, Generator, TransformerConfigs
from labml_nn.transformers.utils import subsequent_mask


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
    heads: int = 8
    dropout: float = 0.0
    d_ff: int = 2048
    n_layers: int = 6


@option(Configs.model)
def feedback_transformer(c: Configs):
    """
    Create [original feedback transformer](index.html).
    """
    from labml_nn.transformers.feedback import FeedbackTransformer, FeedbackTransformerLayer, \
        FeedbackAttention, FeedForward

    return AutoregressiveModel(
        c.n_tokens, c.d_model,
        FeedbackTransformer(
            FeedbackTransformerLayer(d_model=c.d_model,
                                     attn=FeedbackAttention(c.heads, c.d_model, c.dropout),
                                     feed_forward=FeedForward(c.d_model, c.d_ff, c.dropout),
                                     dropout_prob=c.dropout),
            c.n_layers)).to(c.device)


@option(Configs.model)
def feedback_transformer_kv(c: Configs):
    """
    Create [updated feedback transformer](index.html#kv_shared), with precalculated keys and values.
    """
    from labml_nn.transformers.feedback import FeedbackTransformerKV, FeedbackTransformerLayer, \
        FeedbackAttention, FeedForward

    return AutoregressiveModel(
        c.n_tokens, c.d_model,
        FeedbackTransformerKV(
            FeedbackTransformerLayer(d_model=c.d_model,
                                     attn=FeedbackAttention(c.heads, c.d_model, c.dropout,
                                                            is_kv_precomputed=True),
                                     feed_forward=FeedForward(c.d_model, c.d_ff, c.dropout),
                                     dropout_prob=c.dropout),
            c.n_layers, c.d_model, c.heads)).to(c.device)


def main():
    # Create experiment
    experiment.create(name="feedback_transformer")
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

                        # Use `feedback_transformer` for original feedback transformer
                        'model': 'feedback_transformer_kv',

                        'train_loader': 'shuffled_train_loader',
                        'valid_loader': 'shuffled_valid_loader',

                        'seq_len': 128,
                        'epochs': 128,
                        'batch_size': 64,
                        'inner_iterations': 25})

    # Set models for saving and loading
    experiment.add_pytorch_models(get_modules(conf))

    # Start the experiment
    with experiment.start():
        # Run the training loop
        conf.run()


if __name__ == '__main__':
    main()
