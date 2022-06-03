"""
---
title: Rotary Positional Embeddings (RoPE) Experiment
summary: This experiment trains a transformer model with Rotary Positional Embeddings (RoPE) on tiny Shakespeare dataset.
---

# Rotary Positional Embeddings (RoPE) Experiment

This is an annotated PyTorch experiment to train a transformer model with Rotary Positional Embeddings (RoPE).

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/1cf508e693be11ecacc98de8b38a61fe)
"""

from labml import experiment
from labml.configs import calculate
from labml_nn.transformers import TransformerConfigs
from labml_nn.transformers.rope.experiment import Configs as RoPEConfigs


# ### Rotary PE attention

class Configs(RoPEConfigs):  # , ArithmeticAutoregression):
    pass


def _rotary_value_pe_mha(c: TransformerConfigs):
    from labml_nn.transformers.rope.value_pe import RotaryValuePEMultiHeadAttention
    return RotaryValuePEMultiHeadAttention(c.n_heads, c.d_model, 1., 1.)


# Configuration options
calculate(TransformerConfigs.encoder_attn, 'rotary_value', _rotary_value_pe_mha)
calculate(TransformerConfigs.decoder_attn, 'rotary_value', _rotary_value_pe_mha)
calculate(TransformerConfigs.decoder_mem_attn, 'rotary_value', _rotary_value_pe_mha)


def main():
    # Create experiment
    experiment.create(name="rotary_shakespeare", comment="rotary value", writers={'screen', 'labml'})
    # Create configs
    conf = Configs()
    # Override configurations
    experiment.configs(conf, {
        # No fixed positional embeddings
        'transformer.src_embed': 'no_pos',
        'transformer.tgt_embed': 'no_pos',

        # Encoder with RoPE
        'transformer.encoder_attn': 'rotary_value',
        # 'transformer.encoder_attn': 'rotary',

        #
        'model': 'rotary_pe_transformer',

        # Use character level tokenizer
        'tokenizer': 'character',
        # Prompt separator is blank
        'prompt_separator': '',
        # Starting prompt for sampling
        'prompt': 'It is ',
        # Use Tiny Shakespeare dataset
        'text': 'tiny_shakespeare',

        # Use a context size of $256$
        'seq_len': 512,
        # Train for 32 epochs
        'epochs': 24,
        # Batch size $4$
        'batch_size': 16,
        # Switch between training and validation for $10$ times
        # per epoch
        'inner_iterations': 4,

        # Model size
        'd_model': 128,
        'transformer.ffn.d_ff': 512,
        'transformer.n_heads': 4,
        'transformer.dropout': 0.0,

        # Use [Adam optimizer](../../optimizers/noam.html)
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,

        'dataloader_shuffle_with_replacement': True
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
