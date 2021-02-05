"""
---
title: Transformer XL Experiment
summary: This experiment trains a transformer XL model on tiny Shakespeare dataset.
---

# Transformer XL Experiment

This is an annotated PyTorch experiment to train a transformer xl model.
"""
from typing import List

import torch
import torch.nn as nn
from labml.logger import Text

from labml import experiment, tracker, monit, logger
from labml.configs import option
from labml_helpers.metrics.simple_state import SimpleStateModule
from labml_helpers.module import Module
from labml_helpers.train_valid import BatchIndex, hook_model_outputs
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.transformers.xl import TransformerXL, TransformerXLLayer


class AutoregressiveModel(Module):
    """
    ## Auto regressive model
    """

    def __init__(self, n_vocab: int, d_model: int, transformer: TransformerXL):
        super().__init__()
        # Token embedding module
        self.src_embed = nn.Embedding(n_vocab, d_model)
        # Transformer
        self.transformer = transformer
        # Final layer
        self.generator = nn.Linear(d_model, n_vocab)
        self.mask = None

    def forward(self, x: torch.Tensor, mem: List[torch.Tensor]):
        # Initialize the subsequent mask
        length = len(x)
        if mem:
            length += len(mem[0])
        if self.mask is None or self.mask.size(0) != length:
            from labml_nn.transformers.utils import subsequent_mask
            self.mask = subsequent_mask(length).to(x.device)
        # Token embeddings
        x = self.src_embed(x)
        # Run it through the transformer
        res, mem = self.transformer(x, mem, self.mask[:len(x), :length, :])
        # Generate logits of the next token
        res = self.generator(res)
        #
        return res, mem


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    The default configs can and will be over-ridden when we start the experiment
    """

    model: AutoregressiveModel

    # Token embedding size
    d_model: int = 128
    # Number of attention heads
    heads: int = 4
    # Dropout probability
    dropout: float = 0.0
    # Number of features in FFN hidden layer
    d_ff: int = 256
    # Number of transformer layers
    n_layers: int = 6
    #
    memory = SimpleStateModule()

    def init(self):
        # Set tracker configurations
        tracker.set_scalar("accuracy.*", True)
        tracker.set_scalar("loss.*", True)
        # Add a hook to log module outputs
        hook_model_outputs(self.mode, self.model, 'model')
        # Add accuracy as a state module.
        # The name is probably confusing, since it's meant to store
        # states between training and validation for RNNs.
        # This will keep the accuracy metric stats separate for training and validation.
        self.state_modules = [self.accuracy, self.memory]

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        ### Training or validation step
        """

        # Move data to the device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Update global step (number of tokens processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        # Whether to capture model outputs
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # Get model outputs.
            output, mem = self.model(data, self.memory.get())
            self.memory.set(mem)

        # Calculate and cross entropy loss
        loss = self.loss_func(output, target)
        tracker.add("loss.", loss)

        # Calculate and log accuracy
        self.accuracy(output, target)
        self.accuracy.track()

        # Train the model
        if self.mode.is_train:
            # Calculate gradients
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            # Take optimizer step
            self.optimizer.step()
            # Log the model parameters and gradients on last batch of every epoch
            if batch_idx.is_last:
                tracker.add('model', self.model)
            # Clear the gradients
            self.optimizer.zero_grad()

        # Save the tracked metrics
        tracker.save()

    def sample(self):
        """
        ### Sampling function to generate samples periodically while training
        """

        # Starting prompt
        prompt = self.prompt
        # Collect output for printing
        log = [(prompt, Text.subtle)]
        # Sample 25 tokens
        for i in monit.iterate('Sample', 25):
            # Tokenize the prompt
            data = self.text.text_to_i(prompt).unsqueeze(-1)
            data = data.to(self.device)
            # Get the model output
            output, *_ = self.model(data, [])
            # Get the model prediction (greedy)
            output = output.argmax(dim=-1).squeeze()
            # Add the prediction to prompt
            prompt += self.prompt_separator + self.text.itos[output[-1]]
            # Add the prediction for logging
            log += [(self.prompt_separator + self.text.itos[output[-1]], Text.value)]

        # Print the sampled output
        logger.log(log)

@option(Configs.model)
def autoregressive_model(c: Configs):
    """
    ### Initialize the auto-regressive model
    """
    from labml_nn.transformers.xl import RelativeMultiHeadAttention
    from labml_nn.transformers.feed_forward import FeedForward
    m = AutoregressiveModel(c.n_tokens, c.d_model, TransformerXL(
        TransformerXLLayer(d_model=c.d_model,
                           self_attn=RelativeMultiHeadAttention(c.heads, c.d_model, c.dropout),
                           feed_forward=FeedForward(c.d_model, c.d_ff, c.dropout),
                           dropout_prob=c.dropout), c.n_layers))
    return m.to(c.device)


def main():
    """
    ### Run the experiment
    """
    # Create experiment
    experiment.create(name="transformer_xl", comment='')
    # Create configs
    conf = Configs()
    # Load configurations
    experiment.configs(conf,
                       # A dictionary of configurations to override
                       {'tokenizer': 'character',
                        'text': 'tiny_shakespeare',
                        'optimizer.learning_rate': 1.,
                        'optimizer.optimizer': 'Noam',
                        'prompt': 'It is',
                        'prompt_separator': '',

                        'train_loader': 'sequential_train_loader',
                        'valid_loader': 'sequential_valid_loader',

                        'seq_len': 64,
                        'epochs': 128,
                        'batch_size': 32,
                        'inner_iterations': 25,
                        })

    # Set models for saving and loading
    experiment.add_pytorch_models({'model': conf.model})

    # Start the experiment
    with experiment.start():
        # `TrainValidConfigs.run`
        conf.run()


#
if __name__ == '__main__':
    main()
