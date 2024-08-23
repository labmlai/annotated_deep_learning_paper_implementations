"""
---
title: Low-Rank Adaptation (LoRA)
summary: >
  Annotated implementation of RoRA from paper
  LoRA: Low-Rank Adaptation of Large Language Models
---

# Low-Rank Adaptation (LoRA)

This is an implementation of
[Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685)
in [PyTorch](https://pytorch.org).

Low-Rank Adaptation (LoRA) freezes pre-trained model weights and injects
 trainable rank decomposition matrices into each layer of the transformer.
 This makes it possible to efficiently fine-tune large langauge models by
 reducing trainable parameters by a large factor.

Here's [the training code](experiment.html) for training a GPT2 model with LoRA
 on Tiny Shakespeare dataset.
"""

import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    ## LoRA Linear Layer

    LoRA linear layer adds a low-rank decomposition to the pre-trained
    weight matrix ($W_0 \in \mathbb{R}^{d \times k}$)
    of the linear layer.

    $$W_0 + \Delta W = W_0 + BA$$

    , where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$,
     and the rank $r \ll min(d, k)$.

    All parameters are frozen except $A$ and $B$.

    $\Delta W$ is initialized to be zero at the beginning of the training.

    They multiple $x \Delta W^T$ by $\frac{\alpha}{r}$ where $\alpha$ is a hyper-parameter.
    Once $\alpha$ is tuned it can be kept the same when varying $r$.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool,
                 r: int, alpha: int = None):
        """
        :param in_features: is the number of input features of the linear layer
        :param out_features: is the number of output features of the linear layer
        :param bias: is a flag indicating if there is a bias parameter
        :param r: is the rank of the decomposition $r$
        :param alpha: is the scaling factor $\alpha$
        """
        super().__init__()

        # Set $\alpha = r$ is not provided. i.e. make the scaling factor $\frac{\alpha}{r} = 1$.
        if alpha is None:
            alpha = r

        # The pre-trained weight $W_0$
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        # Freeze it
        self.weight.requires_grad = False

        if bias:
            # Bias parameter $b_0$ (also frozen)
            self.bias = nn.Parameter(torch.empty(out_features))
            self.bias.requires_grad = False
        else:
            # No bias parameter
            self.bias = None

        # scaling factor $\frac{\alpha}{r}$
        self.scaling = alpha / r
        # Matrix $A \in \mathbb{R}^{r \times k}$
        self.lora_a = nn.Parameter(torch.empty((r, in_features)))
        # Matrix $B \in \mathbb{R}^{d \times r}$, we keep $A$ and $B$ transposed
        self.lora_b = nn.Parameter(torch.empty((out_features, r)))

        with torch.no_grad():
            # Initialize $A$ similar to a weight matrix in a normal linear layer
            nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
            # Initialize $B$ to $0$ so that $\Delta W = BA$ is $0$ at initialization
            nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor):
        # Compute $x W_0^T + b_0$
        result = nn.functional.linear(x, self.weight, bias=self.bias)

        # Add $\frac{\alpha}{r} x \Delta W^T = \frac{\alpha}{r} x {(BA)}^T = \frac{\alpha}{r} x A^T B^T$
        result += (x @ self.lora_a.T @ self.lora_b.T) * self.scaling

        #
        return result


class Embedding(nn.Module):
    """
    ## LoRA Embedding Layer

    Similar to LoRA linear layer this adds a low-rank decomposition to the pre-trained
    embedding weights matrix ($W_0 \in \mathbb{R}^{d \times k}$).

    $$W_0 + \Delta W = W_0 + BA$$
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 r: int, alpha: int = None):
        """

        :param num_embeddings: is the number of embeddings
        :param embedding_dim: is the number embedding dimensions
        :param r: is the rank of the decomposition $r$
        :param alpha: is the scaling factor $\alpha$
        """
        super().__init__()

        # Set $\alpha = r$ is not provided. i.e. make the scaling factor $\frac{\alpha}{r} = 1$.
        if alpha is None:
            alpha = r

        # The pre-trained embedding weights $W_0^T$ (frozen)
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
        self.weight.requires_grad = False

        # scaling factor $\frac{\alpha}{r}$
        self.scaling = alpha / r
        # Matrix $A \in \mathbb{R}^{r \times k}$
        self.lora_a = nn.Parameter(torch.empty((r, num_embeddings)))
        # Matrix $B \in \mathbb{R}^{d \times r}$
        self.lora_b = nn.Parameter(torch.empty((embedding_dim, r)))

        with torch.no_grad():
            # Initialize $A$ with a normal distribution
            nn.init.normal_(self.lora_a)
            # Initialize $B$ to $0$ so that $\Delta W = BA$ is $0$ at initialization
            nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor):
        # Compute the embeddings $\text{onehot}(x) W_0$
        result = nn.functional.embedding(x, self.weight)

        # Add $\frac{\alpha}{r} \text{onehot}(x) \Delta W^T = \frac{\alpha}{r} \text{onehot}(x) A^T B^T$
        result += (nn.functional.embedding(x, self.lora_a.T) @ self.lora_b.T) * self.scaling

        #
        return result
