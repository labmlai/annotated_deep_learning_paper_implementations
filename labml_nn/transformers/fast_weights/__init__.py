"""
---
title: Linear Transformers Are Secretly Fast Weight Memory Systems
summary: >
  This is an annotated implementation/tutorial of
  Linear Transformers Are Secretly Fast Weight Memory Systems in PyTorch.
---

# Fast weights transformer

The paper
[Linear Transformers Are Secretly Fast Weight Memory Systems in PyTorch](https://arxiv.org/abs/2102.11174)
finds similarities between linear self-attention and fast weight systems
and makes modifications to self-attention update rule based on that.
It also introduces a simpler, yet effective kernel function.

*The authors have provided an [official implementation](https://github.com/ischlag/fast-weight-transformers)
of the paper including other variants they compare with in the paper.*

## Fast weights

Consider a sequence of inputs $\x08ig\\{x^{(i)}\x08ig\\}^L_{i=1}$ or length $L$
and each step is a vector of size $d_{in}$; i.e. $x \\in \\mathbb{R}^{d_{in}}$.
The fast weight model generates a weight matrix at each step to produce output
$\x08ig\\{y^{(i)}\x08ig\\}^L_{i=1}$, $y \\in \\mathbb{R}^{d_{out}}$

\x08egin{align}
a^{(i)}, b^{(i)} &= 	extcolor{orange}{W_a} x^{(i)}, 	extcolor{orange}{W_b} x^{(i)} \\
	extcolor{cyan}{W^{(i)}} &= \\sigma \\Big( 	extcolor{cyan}{W^{(i-1)}} + a^{(i)} \\otimes b^{(i)} \\Big) \\
y^{(i)} &= 	extcolor{cyan}{W^{(i)}} x^{(i)}
\\end{align}

$\\otimes$ is the outer product ($a \\otimes b = a b^	op$), where elements of the two vectors are multiplied with each other
to give a matrix.
$\\sigma$ is an activation function.
$	extcolor{orange}{W_a}$ and $	extcolor{orange}{W_b}$ are trainable weights (parameters).
$	extcolor{cyan}{W^{(i)}}$ are the fast weights that are generated at each step.

## Linear self-attention

Original transformer self-attention is, (omitting $\x0crac{1}{d_k}$ for clarity)

\x08egin{align}
y^{(i)} &= \\Big[v^{(1)}, v^{(2)}, ..., v^{(i)}\\Big] 	ext{softmax}
 \x08igg(
    \\Big[k^{(1)}, k^{(2)}, ..., k^{(i)}\\Big] ^	op
    q^{(i)}
 \x08igg) \\
 &= \\sum^i_{j=1} \x0crac
 { v^{(j)} \\kappa(k^{(j)}, q^{(i)}) }
 { \\sum^i_{j'=1} \\kappa(k^{(j')}, q^{(i)}) } \\
\\end{align}

where $\\kappa(k, q) = 	ext{exp}(k \\cdot q)$

The idea behind linearizing self attention is to replace softmax
kernel $\\kappa$ with a different kernel $\\kappa '$ so that we can calculate the
denominator of the self attention function faster:

$$\\kappa '(k, q) = 	extcolor{lightgreen}{\\phi(k)}^	op 	extcolor{lightgreen}{\\phi(q)}$$

This gives

\x08egin{align}
y^{(i)} &= \x0crac
 {\\Big( \\sum^i_{j=1} v^{(j)} \\otimes 	extcolor{lightgreen}{\\phi(k^{(j)})} \\Big)
  	extcolor{lightgreen}{\\phi(q^{(i)})} }
 { \\Big( \\sum^i_{j'=1}
   	extcolor{lightgreen}{\\phi(k^{(j')})} \\Big)
    	extcolor{lightgreen}{\\phi(q^{(i)})} }
\\end{align}

With $	extcolor{cyan}{W^{(i)}} = \\sum^i_{j=1} v^{(j)} \\otimes \\phi(k^{(j)})$ and
$z^{(i)} = \\sum^i_{j=1} 	extcolor{lightgreen}{\\phi(k^{(j)})}$, we can calculate them efficiently:

\x08egin{align}
	extcolor{cyan}{W^{(i)}} &= 	extcolor{cyan}{W^{(i-1)}} + v^{(i)} \\otimes 	extcolor{lightgreen}{\\phi(k^{(i)})} \\
z^{(i)} &= z{(i)} + 	extcolor{lightgreen}{\\phi(k^{(i)})} \\
y^{(i)} &= \x0crac{1}{z^{(i)} \\cdot 	extcolor{lightgreen}{\\phi(q^{(i)})}}
    W^{(i)} 	extcolor{lightgreen}{\\phi(q^{(i)})}
\\end{align}

This is quite similar to fast weights.

The paper introduces a new linear attention projection function $	extcolor{lightgreen}{\\phi}$
a new update rule for $	extcolor{cyan}{W^{(i)}} = f(	extcolor{cyan}{W^{(i-1)}})$ and change the normalization
$\x0crac{1}{z^{(i)} \\cdot 	extcolor{lightgreen}{\\phi(q^{(i)})}}$

Here are [the training code](experiment.html) and a notebook for training a fast weights
 transformer on the Tiny Shakespeare dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/fast_weights/experiment.ipynb)
"""
import torch
from torch import nn
from labml_helpers.module import Module
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
from labml_nn.utils import clone_module_list

class DPFP(Module):
    """
    ## Deterministic Parameter Free Project (DPFP)

    This is the new projection function $	extcolor{lightgreen}{\\phi}$ introduced in the paper.
    DPFP projects $k$ of dimensionality $d_{key}$ to dimensionality $d_{dot} = 2 d_{key} 
u$,
    where $
u \\in \\{1, 2, ..., 2 d_{key} - 1 \\}$ is a hyper-parameter.

    $$	extcolor{lightgreen}{\\phi_{2 d_{key} (i - 1)  + j}(k)}
     = 	ext{ReLU}\\Big(\x08ig[k, -k\x08ig]\\Big)_{j}
                        	ext{ReLU}\\Big(\x08ig[k, -k\x08ig]\\Big)_{i + j}$$

    where $\x08ig[k, -k\x08ig]$ is the concatenation of $k$ and $-k$ to give a vector of
    size $2 d_{key}$, $i \\in \\{1, 2, ..., 
u \\}$, and $j \\in \\{1, 2, ..., 2 d_{key}\\}$.
    $x_i$ is the $i$-th element of vector $x$ and is rolled around if
    $i$ is larger than the number of elements in $x$.

    Basically, it creates a new vector by multiplying elements of $[k, -k]$ shifted by $i$.

    This produces projections that are sparse (only a few elements of $phi$ are non-zero) and
    orthogonal ($	extcolor{lightgreen}{\\phi(k^{(i)})} \\cdot 	extcolor{lightgreen}{\\phi(k^{(j)})}
     \x07pprox 0$ for most $i, j$
    unless $k^{(i)}$ and $k^{(j)}$ are very similar.

    ### Normalization

    Paper introduces a simple normalization for $	extcolor{lightgreen}{\\phi}$,

    $$	extcolor{lightgreen}{\\phi '(k)} =
     \x0crac{	extcolor{lightgreen}{\\phi(k)}}{\\sum^{d_{dot}}_{j=1} 	extcolor{lightgreen}{\\phi(k)_j}}$$

    *Check the paper for derivation.*
    """

    def __init__(self, nu: int=1, eps: float=1e-06):
        """
        * `nu` is the hyper-parameter $
u$.
        * `eps` is the small value used to make sure there is no division-by-zero when normalizing.
        """
        super().__init__()
        self.nu = nu
        self.relu = nn.ReLU()
        self.eps = eps

    def forward(self, k: torch.Tensor):
        """Implement the forward operation on the given input: k."""
        k = self.dpfp(k)
        return k / (torch.sum(k, dim=-1, keepdim=True) + self.eps)

    def dpfp(self, k: torch.Tensor):
        """
        $$	extcolor{lightgreen}{\\phi(k)}$$
        """
        x = self.relu(torch.cat([k, -k], dim=-1))
        x_rolled = [x.roll(shifts=i, dims=-1) for i in range(1, self.nu + 1)]
        x_rolled = torch.cat(x_rolled, dim=-1)
        x_repeat = torch.cat([x] * self.nu, dim=-1)
        return x_repeat * x_rolled

class FastWeightsAttention(Module):
    """
    ## Fast Weights Attention

    The paper introduces a new update rule for calculating $	extcolor{cyan}{W^{(i)}}$.
    The model first retrieves the current value
    $\x08ar{v}^{(i)}$ paired with the key $k^{(i)}$.
    Then stores a combination $v^{(i)}_{new}$
    of the retrieved value $\x08ar{v}^{(i)}$ and the input $v^{(i)}$.

    \x08egin{align}
    k^{(i)}, v^{(i)}, q^{(i)} &=
     	extcolor{orange}{W_k} x^{(i)}, 	extcolor{orange}{W_v} x^{(i)}, 	extcolor{orange}{W_q} x^{(i)} \\
    \x08ar{v}^{(i)} &= 	extcolor{cyan}{W^{(i-1)}} 	extcolor{lightgreen}{\\phi'(k^{(i)})} \\
    \x08eta^{(i)} &= \\sigma \\Big(	extcolor{orange}{W_\x08eta} x^{(i)} \\Big) \\
    v^{(i)}_{new} &= \x08eta^{(i)} v^{(i)} + \\Big(1 - \x08eta^{(i)} \\Big) \x08ar{v}^{(i)} \\
    	extcolor{cyan}{W^{(i)}}
     &= 	extcolor{cyan}{W^{(i-1)}} + v^{(i)}_{new} \\otimes 	extcolor{lightgreen}{\\phi'(k^{(i)})} \\
     &= 	extcolor{cyan}{W^{(i-1)}} +
     \x08eta^{(i)} \\Big( v^{(i)} - \x08ar{v}^{(i)} \\Big ) \\otimes 	extcolor{lightgreen}{\\phi'(k^{(i)})} \\
    y^{(i)} &= 	extcolor{cyan}{W^{(i)}} 	extcolor{lightgreen}{\\phi'(q^{(i)})}
    \\end{align}

    where $	extcolor{orange}{W_\x08eta}$ is a trainable parameter and $\\sigma$ is the sigmoid function.

    Note that we don't need the normalization term $z$ because $	extcolor{lightgreen}{\\phi'}$ is normalized.
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float, phi: DPFP):
        """Implement the __init__ operation on the given input: heads, d_model, dropout_prob, phi."""
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)
        self.interpolation_weight = nn.Sequential(PrepareForMultiHeadAttention(d_model, heads, 1, bias=False), nn.Sigmoid())
        self.phi = phi
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor):
        """Calculate and return the output of forward based on x."""
        seq_len = x.shape[0]
        query = self.phi(self.query(x))
        key = self.phi(self.key(x))
        value = self.value(x)
        beta = self.interpolation_weight(x)
        weights = key.new_zeros((key.shape[1], key.shape[2], value.shape[3], key.shape[3]))
        outputs = []
        for i in range(seq_len):
            value_existing = torch.einsum('bhvk,bhk->bhv', weights, key[i])
            weights = weights + torch.einsum('bhv,bhk->bhvk', beta[i] * (value[i] - value_existing), key[i])
            y = torch.einsum('bhvk,bhk->bhv', weights, query[i])
            outputs.append(y.reshape(y.shape[0], -1))
        x = torch.stack(outputs)
        return self.output(x)

class FastWeightsAttentionTransformerLayer(Module):
    """
    This is a general transformer layer that combines self attention and feedforward network.
    """

    def __init__(self, *, d_model: int, attn: FastWeightsAttention, feed_forward: FeedForward, dropout_prob: float):
        """Process data using __init__ with arguments no arguments."""
        super().__init__()
        self.size = d_model
        self.attn = attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def forward(self, x: torch.Tensor):
        """Compute forward given x."""
        attn = self.attn(x)
        x = x + self.dropout(attn)
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)
        return x

class FastWeightsAttentionTransformer(Module):
    """
    This is a general transformer module with multiple transformer layers
    """

    def __init__(self, layer: FastWeightsAttentionTransformerLayer, n_layers: int):
        """Process data using __init__ with arguments layer, n_layers."""
        super().__init__()
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor):
        """Execute forward with input (x)."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return self.norm(x)