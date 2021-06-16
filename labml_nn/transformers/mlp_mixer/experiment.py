"""
---
title: MLP Mixer experiment
summary: This experiment trains MLP Mixer on Tiny Shakespeare dataset.
---

# [MLP Mixer](index.html) Experiment

This is an annotated PyTorch experiment to train a [MLP Mixer Model](index.html).
"""

from labml import experiment
from labml.configs import option
from labml_nn.transformers import TransformerConfigs
from labml_nn.transformers.configs import FeedForwardConfigs
from labml_nn.transformers.mlm.experiment import TransformerMLM, Configs as MLMConfigs


class Configs(MLMConfigs):
    """
    ## Configurations

    This inherits from
    [`MLMConfigs`](../mlm/experiment.html) where we define an experiment for
    [Masked Language Models](../mlm.index.html).
    """

    # Configurable [Feed-Forward Network](../feed_forward.html) for the MLP
    mix_mlp: FeedForwardConfigs


@option(Configs.mix_mlp)
def _mix_mlp_configs(c: Configs):
    """
    The mixing MLP configurations
    """

    conf = FeedForwardConfigs()
    # Size of the MLP is the sequence length, because it is applied across tokens
    conf.d_model = c.seq_len
    # The paper suggests $GELU$ activation
    conf.activation = 'GELU'

    #
    return conf


@option(Configs.transformer)
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
    # Embedding size
    conf.d_model = c.d_model
    # Change attention module to [MLPMixer](index.html)
    from labml_nn.transformers.mlp_mixer import MLPMixer
    conf.encoder_attn = MLPMixer(c.mix_mlp.ffn)

    #
    return conf


def main():
    # Create experiment
    experiment.create(name="mlp_mixer_mlm")
    # Create configs
    conf = Configs()
    # Override configurations
    experiment.configs(conf, {
        # Batch size
        'batch_size': 64,
        # Sequence length of $32$. We use a short sequence length to train faster.
        # Otherwise MLM models take forever to train.
        'seq_len': 32,

        # Train for 1024 epochs.
        'epochs': 1024,
        # Switch between training and validation for $1$ times
        # per epoch
        'inner_iterations': 1,

        # Transformer configurations
        'd_model': 128,
        'transformer.ffn.d_ff': 256,
        'transformer.n_heads': 8,
        'transformer.n_layers': 6,
        'transformer.ffn.activation': 'GELU',

        # Mixer MLP hidden layer size
        'mix_mlp.d_ff': 128,

        # Use [Noam optimizer](../../optimizers/noam.html)
        'optimizer.optimizer': 'Noam',
        'optimizer.learning_rate': 1.,
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
