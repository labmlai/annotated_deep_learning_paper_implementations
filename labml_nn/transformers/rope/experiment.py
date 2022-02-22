"""
---
title: Transformer XL Experiment
summary: This experiment trains a transformer XL model on tiny Shakespeare dataset.
---

# Transformer XL Experiment

This is an annotated PyTorch experiment to train a transformer xl model.
"""

from labml import experiment
from labml.configs import option, calculate
from labml_nn.transformers import TransformerConfigs
from labml_nn.transformers.basic.autoregressive_experiment import AutoregressiveTransformer, Configs


# ### Rotary PE attention
def _rotary_pe_mha(c: TransformerConfigs):
    from labml_nn.transformers.rope import RotaryPEMultiHeadAttention
    return RotaryPEMultiHeadAttention(c.n_heads, c.d_model)


calculate(TransformerConfigs.encoder_attn, 'rotary', _rotary_pe_mha)
calculate(TransformerConfigs.decoder_attn, 'rotary', _rotary_pe_mha)
calculate(TransformerConfigs.decoder_mem_attn, 'rotary', _rotary_pe_mha)


@option(Configs.model, 'rotary_pe_transformer')
def _model(c: Configs):
    """
    Create GPT model and initialize weights
    """
    m = AutoregressiveTransformer(c.transformer.encoder,
                                  c.transformer.src_embed,
                                  c.transformer.generator).to(c.device)

    return m


def main():
    # Create experiment
    experiment.create(name="transformer")
    # Create configs
    conf = Configs()
    # Override configurations
    experiment.configs(conf, {
        'device.cuda_device': 1,

        'transformer.src_embed': 'no_pos',
        'transformer.tgt_embed': 'no_pos',

        'transformer.encoder_attn': 'rotary',

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
        'seq_len': 256,
        # Train for $128$ epochs
        'epochs': 128,
        # Batch size $32$
        'batch_size': 32,
        # Switch between training and validation for $10$ times
        # per epoch
        'inner_iterations': 10,

        # Model size
        'd_model': 128,
        'transformer.ffn.d_ff': 256,

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
