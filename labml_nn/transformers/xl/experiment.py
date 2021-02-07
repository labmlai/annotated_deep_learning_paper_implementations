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
        # Masks
        self.mask_x = None
        self.mask_mem = None

    def forward(self, x: torch.Tensor, mem: List[torch.Tensor]):
        # Length of the memory
        m_len = len(mem[0]) if mem else 0
        # Create a subsequent mask for tokens
        if self.mask_x is None or self.mask_x.shape[0] < len(x):
            from labml_nn.transformers.utils import subsequent_mask
            self.mask_x = subsequent_mask(len(x)).to(x.device)
        # Create an all ones (full visibility) mask for memory
        if self.mask_mem is None or self.mask_mem.shape[1] < m_len or self.mask_mem.shape[0] < len(x):
            self.mask_mem = self.mask_x.new_ones(len(x), m_len, 1)

        # Concatenate the masks if there is memory
        if m_len:
            mask = torch.cat((self.mask_mem[:len(x), :m_len], self.mask_x[:len(x), :len(x)]), dim=1)
        # Use the subsequent mask otherwise
        else:
            mask = self.mask_x[:len(x), :len(x)]

        # Token embeddings
        x = self.src_embed(x)
        # Run it through the transformer
        res, mem = self.transformer(x, mem, mask)
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
    # Number of memories to keep
    mem_len: int = 128
    # State module to maintain memories when switching between training and validation
    memory = SimpleStateModule()

    def init(self):
        # Set tracker configurations
        tracker.set_scalar("accuracy.*", True)
        tracker.set_scalar("loss.*", True)
        # Add a hook to log module outputs
        hook_model_outputs(self.mode, self.model, 'model')
        # This will keep the accuracy metric stats and memories separate for training and validation.
        self.state_modules = [self.accuracy, self.memory]

    def merge_memory(self, old_mem, new_mem):
        """
        Concatenate memories and remove old memories to keep a maximum of
        `mem_len` memories.
        """

        # If it's configured not to use memory
        if self.mem_len == 0:
            return []

        # Concatenate with old memory
        if old_mem:
            mem = [torch.cat((m, x), dim=0) for m, x in zip(old_mem, new_mem)]
        else:
            mem = new_mem

        # Truncate old memories
        if len(mem[0]) > self.mem_len:
            mem = [m[-self.mem_len:] for m in mem]

        #
        return mem

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        ### Training/validation step
        """

        # Move data to the device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Update global step (number of tokens processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        # Whether to capture model outputs
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # Get memories
            mem = self.memory.get()
            # Run the model
            output, new_mem = self.model(data, mem)
            # Merge memory
            mem = self.merge_memory(mem, new_mem)
            # Update memories
            self.memory.set(mem)

        # Calculate and log cross entropy loss
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
        # memory
        mem = []
        # Sample 25 tokens
        for i in monit.iterate('Sample', 25):
            # Tokenize the prompt
            data = self.text.text_to_i(prompt).unsqueeze(-1)
            # Move to device
            data = data.to(self.device)
            # Get the model output
            output, new_mem = self.model(data, mem)
            # Get the model prediction (greedy)
            output = output.argmax(dim=-1).squeeze(1)
            # Add the prediction to prompt
            prompt += self.prompt_separator + self.text.itos[output[-1]]
            # Only feed the last character to model in next iteration, rest will go in as memories
            prompt = prompt[-1:]
            # Add the prediction for logging
            log += [(self.prompt_separator + self.text.itos[output[-1]], Text.value)]
            # Update memory
            mem = self.merge_memory(mem, new_mem)

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

                        'seq_len': 2,
                        'mem_len': 32,
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
