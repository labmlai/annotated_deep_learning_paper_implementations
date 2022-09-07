"""
---
title: DeepNorm
summary: >
 A PyTorch implementation/tutorial of DeepNorm from paper DeepNet: Scaling Transformers to 1,000 Layers.
---

# DeepNorm

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/normalization/deep_norm/experiment.ipynb)

This is a [PyTorch](https://pytorch.org) implementation of
the DeepNorm from the paper
[DeepNet: Scaling Transformers to 1,000 Layers](https://papers.labml.ai/paper/2203.00555).

The paper proposes a method to stabilize extremely deep transformers through a new normalizing function
to replace LayerNorm and a weight initialization scheme.
This combines the performance of Post-LayerNorm and the stability of Pre-LayerNorm.
Transformers with DeepNorms are supposed to be stable even without a learning rate warm-up.

The paper first shows that the changes to layer outputs (for the same input)
 change gradually during stable training;
when unstable it changes rapidly during the initial training steps.
This happens with initializing weights to small values, and learning rate warm-ups where the
training is stable.
They use the idea of keeping the changes to layer outputs small to derive the new
 normalization and weight initialization mechanism.

## Weight Initializations

Usually, the weights are initialized with Xavier or Kaiming initializations.
This paper scales (sets the gain) the weights by a constant $\beta$ depending on the size of the
 transformer.

DeepNorm suggests scaling the weights of the two linear transforms in the
[Feed-Forward Network](../../transformers/feed_forward.html),
the value projection transform, and the output projection transform of the
attention layer.
Weights of these transforms are scaled by (has a gain equal to) $\beta$.

The scaling is implemented in the

## Normalization Function

$$x_{l + 1} = \mathop{LN}\Big( \alpha x_l + \mathop{G}_l \big(x_l, \theta_l \big)\Big)$$

where $\alpha$ is a constant that depends on the depth of the transformer,
 $\mathop{LN}$ is [Layer Normalization](../layer_norm/index.html), and
 $\mathop{G}_l (x_l, \theta_l)$ is the function of the $l$-th transformer sub-layer (FFN or attention).

This function is used to replace Post-LayerNorm.

## $\alpha$ and $\beta$ constants

\begin{align}
\begin{array} {c|cc|cc}
\text{Type} & \text{Enc-} \alpha & \text{Enc-} \beta &  \text{Dec-} \alpha & \text{Dec-} \beta \\
\hline \\
\text{Encoder only} & (2N)^{\frac{1}{4}} & (8N)^{-\frac{1}{4}} & - & - \\
\text{Decoder only} & - & - & (2M)^{\frac{1}{4}} & (8M)^{-\frac{1}{4}} \\
\text{Enc-Dec} & 0.81 (N^4M)^{\frac{1}{16}} & 0.87 (N^4 M)^{-\frac{1}{16}} &
 (3M)^{\frac{1}{4}} & (12M)^{-\frac{1}{4}} \\
\end{array}
\end{align}

Where $N$ is the number of layers in the encoder and $M$ is the number of layers in the decoder.

Refer to [the paper](https://papers.labml.ai/paper/2203.00555) for derivation.

[Here is an experiment implementation](experiment.html) that uses DeepNorm.
"""

from typing import Union, List

import torch
from torch import nn, Size

from labml_nn.normalization.layer_norm import LayerNorm
from labml_nn.transformers import MultiHeadAttention
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.utils import subsequent_mask


class DeepNorm(nn.Module):
    """
    ## DeepNorm Normalization

    $$x_{l + 1} = \mathop{LN}\Big( \alpha x_l + \mathop{G}_l \big(x_l, \theta_l \big)\Big)$$
    """

    def __init__(self, alpha: float, normalized_shape: Union[int, List[int], Size], *,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True):
        """
        :param alpha: is $\alpha$
        :param normalized_shape: is the shape for LayerNorm $\mathop{LN}$
        :param eps: is $\epsilon$ for LayerNorm
        :param elementwise_affine: is a flag indicating whether to do an elementwise transformation in LayerNorm
        """
        super().__init__()

        self.alpha = alpha
        # Initialize $\mathop{LN}$
        self.layer_norm = LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor, gx: torch.Tensor):
        """
        :param x: is the output from the previous layer $x_l$
        :param gx: is the output of the current sub-layer $\mathop{G}_l (x_l, \theta_l)$
        """
        # $$x_{l + 1} = \mathop{LN}\Big( \alpha x_l + \mathop{G}_l \big(x_l, \theta_l \big)\Big)$$
        return self.layer_norm(x + self.alpha * gx)


class DeepNormTransformerLayer(nn.Module):
    """
    ## Transformer Decoder Layer with DeepNorm

    This implements a transformer decoder layer with DeepNorm.
    Encoder layers will have a similar form.
    """
    def __init__(self, *,
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 feed_forward: FeedForward,
                 deep_norm_alpha: float,
                 deep_norm_beta: float,
                 ):
        """
        :param d_model: is the token embedding size
        :param self_attn: is the self attention module
        :param feed_forward: is the feed forward module
        :param deep_norm_alpha: is $\alpha$ coefficient in DeepNorm
        :param deep_norm_beta: is $\beta$ constant for scaling weights initialization
        """
        super().__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # DeepNorms after attention and feed forward network
        self.self_attn_norm = DeepNorm(deep_norm_alpha, [d_model])
        self.feed_forward_norm = DeepNorm(deep_norm_alpha, [d_model])

        # Scale weights after initialization
        with torch.no_grad():
            # Feed forward network linear transformations
            feed_forward.layer1.weight *= deep_norm_beta
            feed_forward.layer2.weight *= deep_norm_beta

            # Attention value projection
            self_attn.value.linear.weight *= deep_norm_beta
            # Attention output project
            self_attn.output.weight *= deep_norm_beta

        # The mask will be initialized on the first call
        self.mask = None

    def forward(self, x: torch.Tensor):
        """
        :param x: are the embeddings of shape `[seq_len, batch_size, d_model]`
        """
        # Create causal mask
        if self.mask is None or self.mask.size(0) != len(x):
            # Subsequent mask, will mask out tokens from seeing future tokens
            self.mask = subsequent_mask(len(x)).to(x.device)

        # Run through self attention, i.e. keys and values are from self
        x = self.self_attn_norm(x, self.self_attn(query=x, key=x, value=x, mask=self.mask))
        # Pass through the feed-forward network
        x = self.feed_forward_norm(x, self.feed_forward(x))

        #
        return x
