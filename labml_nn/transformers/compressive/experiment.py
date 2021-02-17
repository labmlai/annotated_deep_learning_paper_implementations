"""
---
title: Compressive Transformer Experiment
summary: This experiment trains a compressive transformer model on tiny Shakespeare dataset.
---

# Compressive Transformer Experiment

This is an annotated PyTorch experiment to train a compressive transformer model.
"""
from typing import List, Tuple, NamedTuple

import torch
import torch.nn as nn

from labml import experiment, tracker, monit, logger
from labml.configs import option
from labml.logger import Text
from labml_helpers.metrics.simple_state import SimpleStateModule
from labml_helpers.module import Module
from labml_helpers.train_valid import BatchIndex, hook_model_outputs
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.transformers.compressive import CompressiveTransformer, AttentionReconstructionLoss, \
    CompressiveTransformerLayer, Conv1dCompression


class CompressedMemory(NamedTuple):
    mem: List[torch.Tensor]
    c_mem: List[torch.Tensor]


class AutoregressiveModel(Module):
    """
    ## Auto regressive model
    """

    def __init__(self, n_vocab: int, d_model: int, transformer: CompressiveTransformer):
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

    def forward(self, x: torch.Tensor, mem: CompressedMemory):
        # Length of the memory
        if mem is not None:
            mem, c_mem = mem.mem, mem.c_mem
        else:
            mem = []
            c_mem = []

        m_len = len(mem[0]) if mem else 0
        if c_mem:
            m_len += len(c_mem[0])

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
        res, mem = self.transformer(x, mem, c_mem, mask)
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
    mem_len: int = 8
    # State module to maintain memories when switching between training and validation
    memory = SimpleStateModule()
    # Attention Reconstruction Loss
    attention_reconstruction_loss: AttentionReconstructionLoss
    # Compression ratio
    compression_ratio: int = 4
    # Compressed memory length
    c_mem_len: int = 128

    def init(self):
        # Set tracker configurations
        tracker.set_scalar("accuracy.*", True)
        tracker.set_scalar("loss.*", True)
        tracker.set_scalar("ar_loss.*", False)
        # Add a hook to log module outputs
        hook_model_outputs(self.mode, self.model, 'model')
        # This will keep the accuracy metric stats and memories separate for training and validation.
        self.state_modules = [self.accuracy, self.memory]

    @torch.no_grad()
    def merge_memory(self, mem: CompressedMemory, new_mem: List[torch.Tensor]) \
            -> Tuple[CompressedMemory, List[torch.Tensor]]:
        """
        Concatenate memories and remove old memories to keep a maximum of
        `mem_len` memories.
        """

        # If it's configured not to use memory
        if self.mem_len == 0:
            return CompressedMemory([], []), []

        if mem is not None:
            mem, c_mem = mem.mem, mem.c_mem
        else:
            mem, c_mem = [], []
        # Concatenate with old memory
        if mem:
            mem = [torch.cat((m, x), dim=0) for m, x in zip(mem, new_mem)]
        else:
            mem = new_mem

        if len(mem[0]) > self.mem_len:
            n_c_mem = (len(mem[0]) - self.mem_len + self.compression_ratio - 1) // self.compression_ratio
            old_mem = []
            trunc_mem = []
            for m in mem:
                n_old = n_c_mem * self.compression_ratio
                cm, m = torch.split(m, [n_old, len(m) - n_old])
                old_mem.append(cm)
                trunc_mem.append(m)
            mem = trunc_mem

            new_c_mem = []
            for i, layer in enumerate(self.model.transformer.layers):
                new_c_mem.append(layer.compress(old_mem[i]))

            if c_mem:
                c_mem = [torch.cat((m, nm), dim=0) for m, nm in zip(c_mem, new_c_mem)]
            else:
                c_mem = new_c_mem

            # Truncate old memories
            if len(c_mem[0]) > self.c_mem_len:
                c_mem = [m[-self.c_mem_len:] for m in c_mem]
        else:
            old_mem = []

        #
        return CompressedMemory(mem, c_mem), old_mem

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
            mem, old_mem = self.merge_memory(mem, new_mem)
            # Update memories
            self.memory.set(mem)

        # Calculate and log cross entropy loss
        loss = self.loss_func(output, target)
        tracker.add("loss.", loss)

        if old_mem:
            ar_loss = self.attention_reconstruction_loss(new_mem, old_mem)
            tracker.add("ar_loss.", ar_loss)
            loss = loss + ar_loss

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
        mem = CompressedMemory([], [])
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
            mem, _ = self.merge_memory(mem, new_mem)

        # Print the sampled output
        logger.log(log)


@option(Configs.model)
def autoregressive_model(c: Configs):
    """
    ### Initialize the auto-regressive model
    """
    from labml_nn.transformers.xl import RelativeMultiHeadAttention
    from labml_nn.transformers.feed_forward import FeedForward
    m = AutoregressiveModel(c.n_tokens, c.d_model, CompressiveTransformer(
        CompressiveTransformerLayer(d_model=c.d_model,
                                    self_attn=RelativeMultiHeadAttention(c.heads, c.d_model, c.dropout),
                                    feed_forward=FeedForward(c.d_model, c.d_ff, c.dropout),
                                    dropout_prob=c.dropout,
                                    compress=Conv1dCompression(c.compression_ratio, c.d_model)), c.n_layers))
    return m.to(c.device)


@option(Configs.attention_reconstruction_loss)
def attention_reconstruction_loss(c: Configs):
    """
    ### Initialize the auto-regressive model
    """
    return AttentionReconstructionLoss(c.model.transformer.layers)


def main():
    """
    ### Run the experiment
    """
    # Create experiment
    experiment.create(name="compressive_transformer", comment='')
    # Create configs
    conf = Configs()
    # Load configurations
    experiment.configs(conf,
                       # A dictionary of configurations to override
                       {'tokenizer': 'character',
                        'text': 'tiny_shakespeare',
                        'optimizer.learning_rate': 2.5e-4,
                        'optimizer.optimizer': 'AdamW',
                        'prompt': 'It is',
                        'prompt_separator': '',

                        'train_loader': 'sequential_train_loader',
                        'valid_loader': 'sequential_valid_loader',

                        'seq_len': 8,
                        'mem_len': 8,
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
