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
from typing import List

import torch
from torch import nn

from labml import experiment, tracker, logger
from labml.configs import option
from labml.logger import Text
from labml_helpers.metrics.accuracy import Accuracy
from labml_helpers.module import Module
from labml_helpers.train_valid import BatchIndex
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.transformers import Encoder, Generator
from labml_nn.transformers import TransformerConfigs
from labml_nn.transformers.mlm import MLM


class TransformerMLM(nn.Module):
    """
    # Transformer based classifier model
    """

    def __init__(self, *, encoder: Encoder, src_embed: Module, generator: Generator):
        """
        * `encoder` is the transformer [Encoder](../models.html#Encoder)
        * `src_embed` is the token
        [embedding module (with positional encodings)](../models.html#EmbeddingsWithLearnedPositionalEncoding)
        * `generator` is the [final fully connected layer](../models.html#Generator) that gives the logits.
        """
        super().__init__()
        self.generator = generator
        self.src_embed = src_embed
        self.encoder = encoder

    def forward(self, x: torch.Tensor):
        # Get the token embeddings with positional encodings
        x = self.src_embed(x)
        # Transformer encoder
        x = self.encoder(x, None)

        y = self.generator(x)

        # Return results
        # (second value is for state, since our trainer is used with RNNs also)
        return y, None


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    This inherits from
    [`NLPAutoRegressionConfigs`](../../experiments/nlp_autoregression.html)
    """

    # Classification model
    model: TransformerMLM
    # Transformer
    transformer: TransformerConfigs

    n_tokens: int = 'n_tokens_mlm'
    no_mask_tokens: List[int] = []
    masking_prob: float = 0.15
    randomize_prob: float = 0.1
    no_change_prob: float = 0.1
    mlm: MLM

    mask_token: int
    padding_token: int

    prompt: str = [
        "We are accounted poor citizens, the patricians good.",
        "What authority surfeits on would relieve us: if they",
        "would yield us but the superfluity, while it were",
        "wholesome, we might guess they relieved us humanely;",
        "but they think we are too dear: the leanness that",
        "afflicts us, the object of our misery, is as an",
        "inventory to particularise their abundance; our",
        "sufferance is a gain to them Let us revenge this with",
        "our pikes, ere we become rakes: for the gods know I",
        "speak this in hunger for bread, not in thirst for revenge.",
    ]

    def init(self):
        self.mask_token = self.n_tokens - 1
        self.padding_token = self.n_tokens - 2

        self.mlm = MLM(masking_prob=self.masking_prob,
                       randomize_prob=self.randomize_prob,
                       no_change_prob=self.no_change_prob,
                       padding_token=self.padding_token,
                       mask_token=self.mask_token,
                       no_mask_tokens=self.no_mask_tokens,
                       n_tokens=self.n_tokens)

        self.accuracy = Accuracy(ignore_index=self.padding_token)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.padding_token)
        super().init()

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        ### Training or validation step
        """

        # Move data to the device
        data = batch[0].to(self.device)

        # Update global step (number of tokens processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        with torch.no_grad():
            data, labels = self.mlm(data)

        # Whether to capture model outputs
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # Get model outputs.
            # It's returning a tuple for states when using RNNs.
            # This is not implemented yet.
            output, *_ = self.model(data)

        loss = self.loss_func(output.view(-1, output.shape[-1]), labels.view(-1))
        tracker.add("loss.", loss)

        # Calculate and log accuracy
        self.accuracy(output, labels)
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

    @torch.no_grad()
    def sample(self):
        """
        ### Sampling function to generate samples periodically while training
        """

        data = torch.zeros((self.seq_len, len(self.prompt)), dtype=torch.long)
        for i, p in enumerate(self.prompt):
            d = self.text.text_to_i(p)
            s = min(self.seq_len, len(d))
            data[:s, i] = d[:s]
        data = data.to(self.device)

        data, labels = self.mlm(data)
        output, *_ = self.model(data)

        for j in range(data.shape[1]):
            log = []
            for i in range(len(data)):
                if labels[i, j] != self.padding_token:
                    t = output[i, j].argmax().item()
                    if t < self.padding_token:
                        if t == labels[i, j]:
                            log.append((self.text.itos[t], Text.value))
                        else:
                            log.append((self.text.itos[t], Text.danger))
                    else:
                        log.append(('*', Text.danger))
                elif data[i, j] < self.padding_token:
                    log.append((self.text.itos[data[i, j]], Text.subtle))

            # Print the sampled output
            logger.log(log)


@option(Configs.n_tokens)
def n_tokens_mlm(c: Configs):
    """
    Get number of tokens
    """
    return c.text.n_tokens + 2


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
    conf.d_model = c.d_model

    #
    return conf


@option(Configs.model)
def _model(c: Configs):
    """
    Create classification model
    """
    m = TransformerMLM(encoder=c.transformer.encoder,
                       src_embed=c.transformer.src_embed,
                       generator=c.transformer.generator).to(c.device)

    return m


def main():
    # Create experiment
    experiment.create(name="mlm")
    # Create configs
    conf = Configs()
    # Override configurations
    experiment.configs(conf, {
        'batch_size': 64,
        'seq_len': 32,

        # Train for 1024 epochs
        'epochs': 1024,
        # Switch between training and validation for $10$ times
        # per epoch
        'inner_iterations': 1,

        # Transformer configurations (same as defaults)
        'd_model': 128,
        'transformer.ffn.d_ff': 256,
        'transformer.n_heads': 8,
        'transformer.n_layers': 6,
        # 'transformer.ffn.activation': 'GELU',

        # Use [Noam optimizer](../../optimizers/noam.html)
        'optimizer.optimizer': 'Noam',
        'optimizer.learning_rate': 1.,
        # 'optimizer.optimizer': 'Adam',
        # 'optimizer.learning_rate': 1e-4,
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
