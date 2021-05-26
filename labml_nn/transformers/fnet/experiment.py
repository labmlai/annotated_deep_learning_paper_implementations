"""
---
title: FNet Experiment
summary: This experiment trains a FNet based model on AG News dataset.
---

# [FNet](index.html) Experiment

This is an annotated PyTorch experiment to train a [FNet model](index.html).

This is based on
[general training loop and configurations for AG News classification task](../../experiments/nlp_classification.html).
"""

import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.nlp_classification import NLPClassificationConfigs
from labml_nn.transformers import Encoder
from labml_nn.transformers import TransformerConfigs


class TransformerClassifier(nn.Module):
    def __init__(self, encoder: Encoder, src_embed: Module, generator: nn.Linear):
        """
        * `encoder` is the transformer [Encoder](../models.html#Encoder)
        * `src_embed` is the token
        [embedding module (with positional encodings)](../models.html#EmbeddingsWithLearnedPositionalEncoding)
        * `generator` is the [final fully connected layer](../models.html#Generator) that gives the logits.
        """
        super().__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.generator = generator

    def forward(self, x: torch.Tensor):
        # Get the token embeddings with positional encodings
        x = self.src_embed(x)
        # Transformer encoder
        x = self.encoder(x, None)
        # Get logits
        x = self.generator(x[-1])

        # Return results
        # (second value is for state, since our trainer is used with RNNs also)
        return x, None


class Configs(NLPClassificationConfigs):
    """
    ## Configurations

    This inherits from
    [`NLPAutoRegressionConfigs`](../../experiments/nlp_autoregression.html#NLPAutoRegressionConfigs)
    """

    # GPT model
    model: TransformerClassifier
    # Transformer
    transformer: TransformerConfigs


@option(Configs.transformer, 'GPT')
def _transformer_configs(c: Configs):
    """
    ### Transformer configurations
    """

    # We use our
    # [configurable transformer implementation](../configs.html#TransformerConfigs)
    conf = TransformerConfigs()
    # Set the vocabulary sizes for embeddings and generating logits
    conf.n_src_vocab = c.n_tokens
    conf.n_tgt_vocab = c.n_tokens

    #
    return conf


@option(TransformerConfigs.encoder_attn)
def fnet(c: TransformerConfigs):
    from labml_nn.transformers.fnet import FNet
    return FNet()


@option(Configs.model)
def _model(c: Configs):
    """
    Create GPT model and initialize weights
    """
    m = TransformerClassifier(c.transformer.encoder,
                              c.transformer.src_embed,
                              nn.Linear(c.d_model, c.n_classes)).to(c.device)

    return m


def main():
    # Create experiment
    experiment.create(name="gpt")
    # Create configs
    conf = Configs()
    # Override configurations
    experiment.configs(conf, {
        # Use character level tokenizer
        'tokenizer': 'basic_english',

        # Train for $32$ epochs
        'epochs': 32,
        # Switch between training and validation for $10$ times
        # per epoch
        'inner_iterations': 10,

        # Transformer configurations
        'transformer.d_model': 512,
        'transformer.ffn.d_ff': 2048,
        'transformer.n_heads': 8,
        'transformer.n_layers': 6,

        'transformer.encoder_attn': 'fnet',

        'optimizer.optimizer': 'Noam',
        # 'optimizer.learning_rate': 1.25e-4,

        # 'seq_len': 768,
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
