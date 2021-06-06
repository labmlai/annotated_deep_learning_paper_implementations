"""
---
title: Masked Language Model
summary: >
  This is an annotated implementation/tutorial of the Masked Language Model in PyTorch.
---

# Masked Language Model (MLM)

This is a [PyTorch](https://pytorch.org) implementation of the Masked Language Model (MLM)
 used to pre-train the BERT model introduced in the paper
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).

## BERT Pretraining

BERT model is a transformer model.
The paper pre-trains the model using MLM and with next sentence prediction.
We have only implemented MLM here.

### Next sentence prediction

In *next sentence prediction*, the model is given two sentences `A` and `B` and the model
makes a binary prediction whether `B` is the sentence that follows `A` in the actual text.
The model is fed with actual sentence pairs 50% of the time and random pairs 50% of the time.
This classification is done while applying MLM. *We haven't implemented this here.*

## Masked LM

This masks a percentage of tokens at random and trains the model to predict
the masked tokens.
They **mask 15% of the tokens** by replacing them with a special `[MASK]` token.

The loss is computed on predicting the masked tokens only.
This causes a problem during fine-tuning and actual usage since there are no `[MASK]` tokens
 at that time.
Therefore we might not get any meaningful representations.

To overcome this **10% of the masked tokens are replaced with the original token**,
and another **10% of the masked tokens are replaced with a random token**.
This trains the model to give representations about the actual token whether or not the
input token at that position is a `[MASK]`.
And replacing with a random token causes it to
give a representation that has information from the context as well;
because it has to use the context to fix randomly replaced tokens.

## Training

MLMs are harder to train than autoregressive models because they have a smaller training signal.
i.e. only a small percentage of predictions are trained per sample.

Another problem is since the model is bidirectional, any token can see any other token.
This makes the "credit assignment" harder.
Let's say you have the character level model trying to predict `home *s where i want to be`.
At least during the early stages of the training, it'll be super hard to figure out why the
replacement for `*` should be `i`, it could be anything from the whole sentence.
Whilst, in an autoregressive setting the model will only have to use `h` to predict `o` and
`hom` to predict `e` and so on. So the model will initially start predicting with a shorter context first
and then learn to use longer contexts later.
Since MLMs have this problem it's a lot faster to train if you start with a smaller sequence length
initially and then use a longer sequence length later.

Here is [the training code](experiment.html) for a simple MLM model.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/3a6d22b6c67111ebb03d6764d13a38d1)
"""

from typing import List

import torch


class MLM:
    """
    ## Masked LM (MLM)

    This class implements the masking procedure for a given batch of token sequences.
    """

    def __init__(self, *,
                 padding_token: int, mask_token: int, no_mask_tokens: List[int], n_tokens: int,
                 masking_prob: float = 0.15, randomize_prob: float = 0.1, no_change_prob: float = 0.1,
                 ):
        """
        * `padding_token` is the padding token `[PAD].
          We will use this to mark the labels that shouldn't be used for loss calculation.
        * `mask_token` is the masking token `[MASK]`.
        * `no_mask_tokens` is a list of tokens that should not be masked.
        This is useful if we are training the MLM with another task like classification at the same time,
        and we have tokens such as `[CLS]` that shouldn't be masked.
        * `n_tokens` total number of tokens (used for generating random tokens)
        * `masking_prob` is the masking probability
        * `randomize_prob` is the probability of replacing with a random token
        * `no_change_prob` is the probability of replacing with original token
        """
        self.n_tokens = n_tokens
        self.no_change_prob = no_change_prob
        self.randomize_prob = randomize_prob
        self.masking_prob = masking_prob
        self.no_mask_tokens = no_mask_tokens + [padding_token, mask_token]
        self.padding_token = padding_token
        self.mask_token = mask_token

    def __call__(self, x: torch.Tensor):
        """
        * `x` is the batch of input token sequences.
         It's a tensor of type `long` with shape `[seq_len, batch_size]`.
        """

        # Mask `masking_prob` of tokens
        full_mask = torch.rand(x.shape, device=x.device) < self.masking_prob
        # Unmask `no_mask_tokens`
        for t in self.no_mask_tokens:
            full_mask &= x != t

        # A mask for tokens to be replaced with original tokens
        unchanged = full_mask & (torch.rand(x.shape, device=x.device) < self.no_change_prob)
        # A mask for tokens to be replaced with a random token
        random_token_mask = full_mask & (torch.rand(x.shape, device=x.device) < self.randomize_prob)
        # Indexes of tokens to be replaced with random tokens
        random_token_idx = torch.nonzero(random_token_mask, as_tuple=True)
        # Random tokens for each of the locations
        random_tokens = torch.randint(0, self.n_tokens, (len(random_token_idx[0]),), device=x.device)
        # The final set of tokens that are going to be replaced by `[MASK]`
        mask = full_mask & ~random_token_mask & ~unchanged

        # Make a clone of the input for the labels
        y = x.clone()

        # Replace with `[MASK]` tokens;
        # note that this doesn't include the tokens that will have the original token unchanged and
        # those that get replace with a random token.
        x.masked_fill_(mask, self.mask_token)
        # Assign random tokens
        x[random_token_idx] = random_tokens

        # Assign token `[PAD]` to all the other locations in the labels.
        # The labels equal to `[PAD]` will not be used in the loss.
        y.masked_fill_(~full_mask, self.padding_token)

        # Return the masked input and the labels
        return x, y
