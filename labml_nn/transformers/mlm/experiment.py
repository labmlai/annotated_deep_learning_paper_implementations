"""
---
title: Masked Language Model Experiment
summary: This experiment trains Masked Language Model (MLM) on Tiny Shakespeare dataset.
---

# [Masked Language Model (MLM)](index.html) Experiment

This is an annotated PyTorch experiment to train a [Masked Language Model](index.html).
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
    # Transformer based model for MLM
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
        # Logits for the output
        y = self.generator(x)

        # Return results
        # (second value is for state, since our trainer is used with RNNs also)
        return y, None


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    This inherits from
    [`NLPAutoRegressionConfigs`](../../experiments/nlp_autoregression.html)
    because it has the data pipeline implementations that we reuse here.
    We have implemented a custom training step form MLM.
    """

    # MLM model
    model: TransformerMLM
    # Transformer
    transformer: TransformerConfigs

    # Number of tokens
    n_tokens: int = 'n_tokens_mlm'
    # Tokens that shouldn't be masked
    no_mask_tokens: List[int] = []
    # Probability of masking a token
    masking_prob: float = 0.15
    # Probability of replacing the mask with a random token
    randomize_prob: float = 0.1
    # Probability of replacing the mask with original token
    no_change_prob: float = 0.1
    # [Masked Language Model (MLM) class](index.html) to generate the mask
    mlm: MLM

    # `[MASK]` token
    mask_token: int
    # `[PADDING]` token
    padding_token: int

    # Prompt to sample
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
        """
        ### Initialization
        """

        # `[MASK]` token
        self.mask_token = self.n_tokens - 1
        # `[PAD]` token
        self.padding_token = self.n_tokens - 2

        # [Masked Language Model (MLM) class](index.html) to generate the mask
        self.mlm = MLM(padding_token=self.padding_token,
                       mask_token=self.mask_token,
                       no_mask_tokens=self.no_mask_tokens,
                       n_tokens=self.n_tokens,
                       masking_prob=self.masking_prob,
                       randomize_prob=self.randomize_prob,
                       no_change_prob=self.no_change_prob)

        # Accuracy metric (ignore the labels equal to `[PAD]`)
        self.accuracy = Accuracy(ignore_index=self.padding_token)
        # Cross entropy loss (ignore the labels equal to `[PAD]`)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.padding_token)
        #
        super().init()

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        ### Training or validation step
        """

        # Move the input to the device
        data = batch[0].to(self.device)

        # Update global step (number of tokens processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        # Get the masked input and labels
        with torch.no_grad():
            data, labels = self.mlm(data)

        # Whether to capture model outputs
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # Get model outputs.
            # It's returning a tuple for states when using RNNs.
            # This is not implemented yet.
            output, *_ = self.model(data)

        # Calculate and log the loss
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

        # Empty tensor for data filled with `[PAD]`.
        data = torch.full((self.seq_len, len(self.prompt)), self.padding_token, dtype=torch.long)
        # Add the prompts one by one
        for i, p in enumerate(self.prompt):
            # Get token indexes
            d = self.text.text_to_i(p)
            # Add to the tensor
            s = min(self.seq_len, len(d))
            data[:s, i] = d[:s]
        # Move the tensor to current device
        data = data.to(self.device)

        # Get masked input and labels
        data, labels = self.mlm(data)
        # Get model outputs
        output, *_ = self.model(data)

        # Print the samples generated
        for j in range(data.shape[1]):
            # Collect output from printing
            log = []
            # For each token
            for i in range(len(data)):
                # If the label is not `[PAD]`
                if labels[i, j] != self.padding_token:
                    # Get the prediction
                    t = output[i, j].argmax().item()
                    # If it's a printable character
                    if t < len(self.text.itos):
                        # Correct prediction
                        if t == labels[i, j]:
                            log.append((self.text.itos[t], Text.value))
                        # Incorrect prediction
                        else:
                            log.append((self.text.itos[t], Text.danger))
                    # If it's not a printable character
                    else:
                        log.append(('*', Text.danger))
                # If the label is `[PAD]` (unmasked) print the original.
                elif data[i, j] < len(self.text.itos):
                    log.append((self.text.itos[data[i, j]], Text.subtle))

            # Print
            logger.log(log)


@option(Configs.n_tokens)
def n_tokens_mlm(c: Configs):
    """
    Number of tokens including `[PAD]` and `[MASK]`
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
    # Embedding size
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
        # Batch size
        'batch_size': 64,
        # Sequence length of $32$. We use a short sequence length to train faster.
        # Otherwise it takes forever to train.
        'seq_len': 32,

        # Train for 1024 epochs.
        'epochs': 1024,
        # Switch between training and validation for $1$ times
        # per epoch
        'inner_iterations': 1,

        # Transformer configurations (same as defaults)
        'd_model': 128,
        'transformer.ffn.d_ff': 256,
        'transformer.n_heads': 8,
        'transformer.n_layers': 6,

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
