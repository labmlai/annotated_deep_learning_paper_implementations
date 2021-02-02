"""
---
title: GPT
summary: >
  Implementation/tutorial of GPT model and training code.
---

# GPT

This is a tutorial/implementation of
[OpenAI GPT architecture](https://openai.com/blog/better-language-models/)
in [PyTorch](https://pytorch.org).
We got a bunch of implementation details from
[minGPT](https://github.com/karpathy/minGPT)
by [@karpathy](https://twitter.com/karpathy).
This implementation also uses character tiny shakespeare dataset.

GPT model is essentially a standard transformer with a few tweaks.
GPT-2 and especially GPT-3 models are quite large and won't fit on a
single GPU and will need model parallelism.
This implementation doesn't even use data parallelism and is intended to be
more of a tutorial.

Main differences of this compared to a simple autoregressive transformer
are the parameter initialization, weight decay, and learning rate schedule.
For the transformer we reuse the
[existing labml/nn transformer implementation](../transformers/index.html).

Here's a notebook for training a GPT model on Tiny Shakespeare dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/nn/blob/master/labml_nn/transformers/gpt/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://web.lab-ml.com/run?uuid=0324c6d0562111eba65d0242ac1c0002)
"""

import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.optimizers.configs import OptimizerConfigs
from labml_nn.transformers import TransformerConfigs, Encoder
from labml_nn.transformers.utils import subsequent_mask


class GPT(Module):
    """
    ## GPT model

    This consists of a token embedding layer, transformer encoder, and
    a final linear layer that gives token logits.
    """

    def __init__(self, encoder: Encoder, src_embed: Module, generator: Module):
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

        # The mask will be initialized on the first call
        self.mask = None

    def forward(self, x: torch.Tensor):
        # Create subsequent mask if mask is not initialized
        # or if the size of the mask is different
        if self.mask is None or self.mask.size(0) != len(x):
            # Subsequent mask, will mask out tokens from seeing future tokens
            self.mask = subsequent_mask(len(x)).to(x.device)
        # Get the token embeddings with positional encodings
        x = self.src_embed(x)
        # Transformer encoder
        x = self.encoder(x, self.mask)
        # Get logits
        x = self.generator(x)

        # Return results
        # (second value is for state, since our trainer is used with RNNs also)
        return x, None


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    This inherits from
    [`NLPAutoRegressionConfigs`](../../experiments/nlp_autoregression.html#NLPAutoRegressionConfigs)
    """

    # GPT model
    model: GPT
    # Transformer
    transformer: TransformerConfigs
    # Weight decay
    weight_decay: float = 0.1
    # Number of tokens for wamup
    warmup_steps: int = 128 * 128 * 20

    # Custom optimizer
    optimizer = 'transformer_optimizer'


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
    # GPT uses GELU activation for position wise feedforward
    conf.ffn.activation = 'GELU'

    #
    return conf


def _init_weights(module):
    """
    ### Initialize weights

    Weights of linear layers and embedding layers are initialized
    to $\mathcal{N}(0, 0.02)$
    instead of the default Xavier initialzation.
    """

    if not isinstance(module, (nn.Linear, nn.Embedding)):
        return

    module.weight.data.normal_(mean=0.0, std=0.02)

    # Initialize biases to $0$
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


@option(Configs.model)
def _model(c: Configs):
    """
    Create GPT model and initialize weights
    """
    m = GPT(c.transformer.encoder,
            c.transformer.src_embed,
            c.transformer.generator).to(c.device)

    # Apply custom weight initialization
    m.apply(_init_weights)

    return m


@option(NLPAutoRegressionConfigs.optimizer)
def transformer_optimizer(c: NLPAutoRegressionConfigs):
    """
    ### Create custom optimizer with weight decay

    This code is taken from [minGPT](https://github.com/karpathy/minGPT).
    This applies weight decay only to weights of linear layers.
    """
    # Collect names of parameters to apply weight decay
    decay = set()
    for mn, m in c.model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn  # full param name

            if fpn.endswith('weight') and isinstance(m, nn.Linear):
                decay.add(fpn)

    # Get all the parameters
    param_dict = {pn: p for pn, p in c.model.named_parameters()}
    # Parameters that are not decayed
    no_decay = set(param_dict.keys()) - decay

    # create the pytorch optimizer object
    opt_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": c.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    # Create a [configurable optimizer](../optimizers/configs.html#OptimizerConfigs),
    # so that we can change these simply by passing
    # a config dictionary.
    optimizer = OptimizerConfigs()

    # Set parameter groups for optimization.
    optimizer.parameters = opt_groups
    # Use [cosine decay optimizer](../optimizers/adam_warmup_cosine_decay.html).
    # This is what GPT uses.
    optimizer.optimizer = 'AdamWarmupCosineDecay'
    # Set model embedding size, required if we use [Noam optimizer](../optimizers/noam.html)
    # which has an exponential decay.
    optimizer.d_model = c.d_model
    # Set default weight decay.
    # This is not required since we set the weight decay in the parameter groups.
    optimizer.weight_decay = c.weight_decay
    # GPT uses a maximum learning rate of $6 \times 10^{-4}$.
    optimizer.learning_rate = 6e-4
    # $\beta_1 = 0.9, \beta_2 = 0.95$
    optimizer.betas = (0.9, 0.95)
    # $\epsilon = 10^{-8}$
    optimizer.eps = 1e-8
    # Weight decay is decoupled from gradients
    optimizer.weight_decouple = True
    # Total number of optimization steps for learning rate cosine decay
    optimizer.total_steps = c.epochs * len(c.text.train) // (c.batch_size * c.seq_len)
    # Number of warmup optimization steps
    optimizer.warmup = c.warmup_steps // (c.batch_size * c.seq_len)

    return optimizer


def main():
    # Create experiment
    experiment.create(name="gpt")
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

        # Use a context size of $128$
        'seq_len': 128,
        # Train for $32$ epochs
        'epochs': 32,
        # Batch size $128$
        'batch_size': 128,
        # Switch between training and validation for $10$ times
        # per epoch
        'inner_iterations': 10,

        # Transformer configurations
        'transformer.d_model': 512,
        'transformer.ffn.d_ff': 2048,
        'transformer.n_heads': 8,
        'transformer.n_layers': 6
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
