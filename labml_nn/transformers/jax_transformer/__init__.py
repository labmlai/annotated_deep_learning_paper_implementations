"""
---
title: Autoregressive Transformer Decoder in JAX from scratch
summary: >
    An implementation of a transformer decode on a small text dataset in JAX from scratch,
    with implementations of basic layers like layer normalization and adam optimizer.
---

# Autoregressive Transformer Decoder in JAX from scratch

### Contents

* [Module class to help us write the layers](#Module)
* [Embedding layer](#Embedding)
* [Positional embeddings](#PositionalEmbedding)
* [Linear layer](#Linear)
* [Layer Normalization](#LayerNormalization)
* [Multi-head attention](#MHA)
* [Position-wise Feed-Forward layer](#FFN)
* [TransformerLayer layer](#TransformerLayer)
* [Cross Entropy Loss](#CrossEntropyLoss)
* [Autoregressive Transformer](#AutoregressiveTransformer)
* [Adam Optimizer](#Adam)
* [Simple dataset](#Dataset)
* [Experiment code](#Experiment)
"""

from functools import partial
from typing import Dict, NamedTuple, Tuple, Any, Callable
from typing import List, TypeVar, Generic
from typing import Union, Optional

import jax
import jax.numpy as jnp
import numpy as np

from labml import lab, monit, experiment, tracker
from labml import logger
from labml.logger import Text
from labml.utils.download import download_file


class Module:
    """
    <a id="Module"></a>

    ## Module

    This is a base class for all modules.
    It handles parameters and transforms methods to pure functions for JAX to compile and differentiate.

    You can skip these modules to get into the models directly.

    The modules stores parameters and sub-modules separately.
    When we want to transform any method to a pure function, we pass the parameters of the
    module and the sub-module as an argument and assign the passed values to class.

    This is based on a blog post:
     [From PyTorch to JAX: towards neural net frameworks that purify stateful code](https://sjmielke.com/jax-purify.htm).
    """

    # Store all parameters and sub-modules in dictionaries
    _submodules: Dict[str, 'Module']
    _params: Dict[str, jnp.ndarray]

    def __init__(self):
        """Initialize"""
        self._params = {}
        self._submodules = {}

    def __getattr__(self, attr_name: str):
        """
        ### Get attribute

        We override the get attribute operation. So when you reference
        an attribute with `model.attribute` this function gets called.

        [Read this guide](https://rszalski.github.io/magicmethods/) if you are not familiar with Python
        magic methods.
        """

        # If the attribute is a parameter
        if attr_name in self._params:
            return self._params[attr_name]
        # If the attribute is a sub-module
        elif attr_name in self._submodules:
            return self._submodules[attr_name]
        # Otherwise fallback to normal attributes.
        # The attributes are stored in `__dict__` by Python.
        else:
            return self.__dict__[attr_name]

    def __setattr__(self, key: str, value: Any):
        """
        ### Set attribute

        We override the set attribute operation. So when you assign
        an attribute with `model.attribute` this function gets called.
        """

        # If the value is also a module
        if isinstance(value, Module):
            self._submodules[key] = value
        # If the value is a JAX array
        elif isinstance(value, jnp.ndarray):
            self._params[key] = value
        # Otherwise add it to `__dict__`
        else:
            self.__dict__[key] = value

    def _clear_params(self):
        """
        ### Clear parameters

        These clears out all the parameters. This is used when a method is called as a pure function.
        We first clears out all the parameters and assigns the parameters passed to the pure function.
        """

        # Clear parameters of the module
        self._params = {}
        # Recursively clear parameters of submodules
        for sm in self._submodules.values():
            sm._clear_params()

    def get_params(self) -> Dict[str, jnp.ndarray]:
        """
        ### Collect all the parameters

        This recursively collects all the parameters of the module and sub-modules into a dictionary.
        """

        # Parameters of the model
        params = self._params.copy()
        # Parameters of the submodules
        for sm_name, sm in self._submodules.items():
            for name, value in sm.get_params().items():
                # The dictionary keys are of the form `module_name/module_name/param_name`
                params[sm_name + "/" + name] = value
        #
        return params

    def _set_params(self, params: Dict[str, jnp.ndarray]):
        """
        ### Set all the parameters
        """

        # Iterate through parameters.
        # Their names have the form `module_name/module_name/param_name`
        for name, value in params.items():
            # Split to get module names and parameter name
            self._set_param(name.split("/"), value)

    def _set_param(self, param_path: List[str], value: jnp.ndarray):
        """
        ### Set a single parameter

        This is called by `_set_params`
        """
        # No module names; i.e. a parameter of this module
        if len(param_path) == 1:
            self._params[param_path[0]] = value
        # Parameter of a submodule
        else:
            self._submodules[param_path[0]]._set_param(param_path[1:], value)

    def purify(self, method: Callable) -> Callable:
        """
        ### Transform a member method to a pure function

        This transforms a member method to a pure function that accepts a dictionary of parameters
        as an argument.

        For example,

        ```python
        params = model.get_params()
        pure_function = model.purify(model.calculate_loss)
        output = pure_function(params, data)
        ```
        """

        def pure_method(params: Dict[str, jnp.array], *args):
            # Clear parameters in the object
            self._clear_params()
            # Assign the passed parameters
            self._set_params(params)
            # Invoke the method
            result = method(*args)
            # Return the result
            return result

        #
        return pure_method


# Type for generics in the module list class
M = TypeVar('M', bound=Module)


class ModuleList(Module, Generic[M]):
    """
    ## Module list

    This stores a list of modules.
    We needed this for transformer decoder to hold the list of transformer layers.
    """

    # For list of modules
    _submodules: List[M]

    def __init__(self, modules: List[M]):
        """
        Initialize with a list of modules.
        """
        super().__init__()
        self._submodules = modules

    def __getitem__(self, idx: int) -> M:
        """
        ### Get the `idx`-th module
        """
        return self._submodules[idx]

    def __setitem__(self, key, value):
        """
        This is not supported
        """
        raise NotImplementedError

    def __len__(self):
        """
        ### Number of modules
        """
        return len(self._submodules)

    def __getattr__(self, item):
        """
        Override `__getattr__` of `Module`
        """
        return self.__dict__[item]

    def __setattr__(self, key, value):
        """
        Override `__setattr__` of `Module`
        """
        self.__dict__[key] = value

    def _clear_params(self):
        """
        ### Clear all parameters
        """
        self._params = {}
        for sm in self._submodules:
            sm._clear_params()

    def get_params(self):
        """
        ### Get all parameters
        """
        params = self._params
        for i, sm in enumerate(self._submodules):
            for name, value in sm.get_params().items():
                params[f'{i}/{name}'] = value
        return params

    def _set_param(self, param_path: List[str], value: jnp.ndarray):
        """
        ### Set a parameter
        """
        self._submodules[int(param_path[0])]._set_param(param_path[1:], value)


class Embedding(Module):
    """
    <a id="Embedding"></a>

    ## Embedding layer

    This maintains embeddings by id.
    """

    def __init__(self, rnd_key: jax.random.PRNGKey, n_embeddings: int, n_dim: int):
        """
        * `rnd_key` is the PRNG state
        * `n_embeddings` is the number of embeddings
        * `n_dim` is the size of an embedding
        """
        super().__init__()
        # Embeddings are initialized from $\mathcal{N}(0, 1)$
        self.embeddings = jax.random.normal(rnd_key, (n_embeddings, n_dim))

    def __call__(self, ids: jnp.ndarray):
        """
        Return the embeddings for the given ids
        """
        return self.embeddings[ids, :]


class EmbeddingsWithLearnedPositionalEncoding(Module):
    """
    <a id="PositionalEmbedding"></a>

    ## Embed tokens and add parameterized positional encodings

    This is based on
    [our PyTorch implementation](https://nn.labml.ai/transformers/models.html#EmbeddingsWithLearnedPositionalEncoding).
    """

    def __init__(self, rnd_key: jax.random.PRNGKey, n_vocab: int, d_model: int, max_len: int = 4096):
        """
        * `rnd_key` is the PRNG state
        * `n_vocab` is the vocabulary size
        * `d_model` is the embedding size
        * `max_len` is the maximum sequence length (to initialize positional encodings)
        """
        super().__init__()
        # Embeddings
        self.embeddings = Embedding(rnd_key, n_vocab, d_model)
        # Positional encodings coefficient $\frac{1}{\sqrt{d}}$
        self.pe_coef = 1 / d_model ** 0.5
        # Positional encodings initialized to zeros
        self.positional_encodings = jnp.zeros((max_len, d_model))

    def __call__(self, x: jnp.ndarray):
        # Get positional encodings
        pe = self.positional_encodings[:x.shape[0]]
        # Get embeddings and add positional encodings
        return self.embeddings(x) * self.pe_coef + pe


class Linear(Module):
    """
    <a id="Linear"></a>

    ## Linear Layer

    This is a simple linear layer with a weight matrix and a bias vector
    """

    def __init__(self, rnd_key: jax.random.PRNGKey, in_features: int, out_features: int):
        """
        * `rnd_key` is the PRNG state
        * `in_features` is the number of features in the input
        * `out_features` is the number of features in the output
        """
        super().__init__()
        # Initialize weights to
        # $$\mathcal{U}\Bigg(-\frac{1}{\sqrt{d_{in}}}, \frac{1}{\sqrt{d_{in}}} \Bigg)$$
        rnd_range = 1 / in_features ** 0.5
        self.weight = jax.random.uniform(rnd_key, (in_features, out_features),
                                         minval=-rnd_range, maxval=rnd_range)
        # Initialize the biases to $0$
        self.bias = jnp.zeros((out_features,))

    def __call__(self, x: jnp.ndarray):
        # Multiply by weights and add the bias
        return jnp.matmul(x, self.weight) + self.bias


class LayerNorm(Module):
    r"""
    <a id="LayerNormalization"></a>

    ## Layer Normalization

    This implements the the layer normalization from the paper
    [Layer Normalization](https://papers.labml.ai/paper/1607.06450).

    When input $X \in \mathbb{R}^{L \times C}$ is a sequence of embeddings,
    where $C$ is the number of channels, $L$ is the length of the sequence.
    $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$.
    $$\text{LN}(X) = \gamma
    \frac{X - \underset{C}{\mathbb{E}}[X]}{\sqrt{\underset{C}{Var}[X] + \epsilon}}
    + \beta$$

    This is based on
    [our PyTorch implementation](https://nn.labml.ai/normalization/layer_norm/index.html).
    """

    def __init__(self, normalized_shape: Union[Tuple[int], List[int]], *,
                 eps: float = 1e-5, elementwise_affine: bool = True):
        """
        * `normalized_shape` $S$ is the shape of the elements (except the batch).
         The input should then be
         $X \in \mathbb{R}^{* \times S[0] \times S[1] \times ... \times S[n]}$
        * `eps` is $\epsilon$, used in $\sqrt{Var[X] + \epsilon}$ for numerical stability
        * `elementwise_affine` is whether to scale and shift the normalized value
        """
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.normalized_shape = tuple(normalized_shape)

        # Create parameters for $\gamma$ and $\beta$ for gain and bias
        if elementwise_affine:
            self.gain = jnp.ones(normalized_shape)
            self.bias = jnp.zeros(normalized_shape)

    def __call__(self, x: jnp.ndarray):
        # Sanity check to make sure the shapes match
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]

        # The exes to calculate the mean and variance on
        axes = [-(i + 1) for i in range(len(self.normalized_shape))]
        # Calculate the mean of all elements;
        # i.e. the means for each element $\mathbb{E}[X]$
        mean = x.mean(axis=axes, keepdims=True)
        # Calculate the squared mean of all elements;
        # i.e. the means for each element $\mathbb{E}[X^2]$
        mean_2 = (x ** 2).mean(axis=axes, keepdims=True)
        # Variance of all element $Var[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$
        var = mean_2 - mean ** 2
        # Normalize $$\hat{X} = \frac{X - \mathbb{E}[X]}{\sqrt{Var[X] + \epsilon}}$$
        x_norm = (x - mean) / (var + self.eps) ** 0.5

        # Scale and shift $$\text{LN}(x) = \gamma \hat{X} + \beta$$
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        #
        return x_norm


class MultiHeadAttention(Module):
    r"""
    <a id="MHA"></a>

    ## Multi-Head Attention Module

    This computes scaled multi-headed attention from
    the paper [Attention Is All You Need](https://papers.labml.ai/paper/1706.03762)
    for given `query`, `key` and `value` vectors.

    $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$

    In simple terms, it finds keys that matches the query, and gets the values of
     those keys.

    It uses dot-product of query and key as the indicator of how matching they are.
    Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
    This is done to avoid large dot-product values causing softmax to
    give very small gradients when $d_k$ is large.

    Softmax is calculated along the axis of of the sequence (or time) for keys.

    This is based on
    [our PyTorch implementation](https://nn.labml.ai/transformers/mha.html#MHA).
    """

    def __init__(self, rnd_key: jax.random.PRNGKey, heads: int, d_model: int):
        """
        * `rnd_key` is the PRNG state
        * `heads` is the number of heads.
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        """

        super().__init__()

        # Split the PRNG state
        _, *rnd_keys = jax.random.split(rnd_key, 5)

        # Number of features per head
        self.d_k = d_model // heads
        # Number of heads
        self.heads = heads

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = Linear(rnd_keys[0], d_model, d_model)
        self.key = Linear(rnd_keys[1], d_model, d_model)
        self.value = Linear(rnd_keys[2], d_model, d_model)

        # Output layer
        self.output = Linear(rnd_keys[3], d_model, d_model)
        # Scaling factor before the softmax
        self.scale = 1 / self.d_k ** 0.5

    def __call__(self, *,
                 query: jnp.ndarray,
                 key: jnp.ndarray,
                 value: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None):
        """
        `query`, `key` and `value` are the tensors that store
        collection of *query*, *key* and *value* vectors.
        They have shape `[seq_len, d_model]`.

        `mask` has shape `[seq_len, seq_len]` and
        `mask[i, j]` indicates whether query at position `i` can see key-value at position `j`.
        """

        # Get sequence length
        seq_len = len(query)

        if mask is not None:
            # Check mask shape
            assert mask.shape[0] == query.shape[0]
            assert mask.shape[1] == key.shape[0]

            # Same mask applied to all heads.
            mask = mask[:, :, None]

        # Apply linear transformations
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Reshape to split into heads
        # Input has shape `[seq_len, batch_size, d_model]`.
        # We split the last dimension into `heads` and `d_k`.
        query = query.reshape(*query.shape[:-1], self.heads, self.d_k)
        key = key.reshape(*key.shape[:-1], self.heads, self.d_k)
        value = value.reshape(*value.shape[:-1], self.heads, self.d_k)

        # Compute attention scores $Q K^\top$.
        # This gives a tensor of shape `[seq_len, seq_len, heads]`.
        # $$S_{ijh} = \sum_d Q_{ihd} K_{jhd}$$
        scores = jnp.einsum('ihd,jhd->ijh', query, key)

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= self.scale

        # Apply mask
        if mask is not None:
            scores = scores + (mask == 0) * float('-inf')

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = jax.nn.softmax(scores, axis=1)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
        x = jnp.einsum("ijh,jhd->ihd", attn, value)

        # Concatenate multiple heads
        x = x.reshape(seq_len, -1)

        # Output layer
        return self.output(x)


class FeedForward(Module):
    """
    <a id="FFN"></a>

    ## Position-wise Feed-Forward layer

    This is based on
    [our PyTorch implementation](https://nn.labml.ai/transformers/feed_forward.html).
    """

    def __init__(self, rnd_key: jax.random.PRNGKey, d_model: int, d_ff: int,
                 activation=jax.nn.relu):
        """
        * `rnd_key` is the PRNG state
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `activation` is the activation function $f$
        """
        super().__init__()
        # Split the PRNG state
        _, *rnd_keys = jax.random.split(rnd_key, 5)

        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer1 = Linear(rnd_keys[0], d_model, d_ff)
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer2 = Linear(rnd_keys[1], d_ff, d_model)
        # Activation function $f$
        self.activation = activation

    def __call__(self, x: jnp.ndarray):
        # $f(x W_1 + b_1)$
        x = self.activation(self.layer1(x))
        # $f(x W_1 + b_1) W_2 + b_2$
        return self.layer2(x)


class TransformerLayer(Module):
    """
    <a id="TransformerLayer"></a>

    ## Transformer Layer

    This is a transformer layer with multi-head attention and a position-wise feed-forward layer.
    We use pre-layer layer normalization.
    """

    def __init__(self,
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 feed_forward: FeedForward):
        """
        * `d_model` is the token embedding size
        * `self_attn` is the self attention module
        * `feed_forward` is the feed forward module
        """
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm_self_attn = LayerNorm([d_model])
        self.norm_ff = LayerNorm([d_model])

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray):
        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        # Run through self attention, i.e. keys and values are from self
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        x = x + self_attn

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the feed-forward network
        ff = self.feed_forward(z)
        # Add the feed-forward results
        x = x + ff
        #
        return x


class CrossEntropyLoss(Module):
    """
    <a id="CrossEntropyLoss"></a>

    ## Cross Entropy Loss
    """

    def __init__(self):
        super().__init__()

        # Use `jax.vmap` to vectorize the loss function
        self._loss_vmap = jax.vmap(self._loss, in_axes=(0, 0,))

    def _loss(self, output: jnp.ndarray, target: jnp.ndarray):
        # $$- \sum_k y_k \log \hat{y}_k$$
        return -jax.nn.log_softmax(output)[target]

    def __call__(self, output: jnp.ndarray, target: jnp.ndarray):
        """
        * `output` is the model outputs of shape `[seq_len, n_vocab]`
        * `target` is the target of shape `[seq_len]`
        """

        # Use the vectorized loss function and calculate the mean.
        #
        # We could have used a for loop to calculate the losses but using vmap is about 10X faster
        return self._loss_vmap(output, target).mean()


class AutoregressiveTransformer(Module):
    """
    <a id="AutoregressiveTransformer"></a>

    ## Autoregressive Transformer

    This is the transformer decode with embedding and output layers.
    """
    layers: ModuleList[TransformerLayer]

    def __init__(self, rnd_key: jax.random.PRNGKey, n_vocab: int, d_model: int, n_layers: int, heads: int, d_ff: int):
        """
        * `rnd_key` is the PRNG state
        * `n_vocab` is the vocabulary size
        * `d_model` is the number of features in a token embedding
        * `n_layers` is the number of transformer layers
        * `heads` is the number of attention heads
        * `d_ff` is the number of features in the hidden layer of the FFN
        """
        super().__init__()
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.loss_func = CrossEntropyLoss()

        # For transformer layers
        layers = []
        for i in range(n_layers):
            # Split PRNG state
            rnd_key, mha_key, ffn_key = jax.random.split(rnd_key, 3)
            # Create a transformer layer
            attn = MultiHeadAttention(mha_key, heads, d_model)
            ffn = FeedForward(ffn_key, d_model, d_ff)
            layers.append(TransformerLayer(d_model, attn, ffn))
        # Make a module list
        self.layers = ModuleList(layers)

        # Split PRNG state
        rnd_key, emb_key, out_key = jax.random.split(rnd_key, 3)
        # Create embedding layer
        self.embeddings = EmbeddingsWithLearnedPositionalEncoding(emb_key, n_vocab, d_model)
        # Final normalization and output layer
        self.norm = LayerNorm([d_model])
        self.output = Linear(out_key, d_model, n_vocab)

    def __call__(self, x: jnp.ndarray):
        # Get sequence length
        seq_len = len(x)
        # A mask for attention so that a token can only see tokens before that
        mask = jnp.tril(jnp.ones((seq_len, seq_len), bool))
        # Get embeddings with positional encodings
        x = self.embeddings(x)
        # Apply the transformer layers
        for i in range(len(self.layers)):
            x = self.layers[i](x, mask)

        # Final normalization and linear transformation to get the logits
        return self.output(self.norm(x))

    def get_loss(self, x: jnp.ndarray):
        """
        ### Calculate the loss
        """
        # Get model outputs
        output = self(x)
        # Cross entropy loss
        return self.loss_func(output[:-1], x[1:])

    def sample(self, seq: jnp.ndarray, length: int = 20):
        """
        ### Sample

        The starting sequence is given by `seq` and we greedily sample `length1 tokens
        """
        for i in range(length):
            # Sample the highest probability token
            idx = jnp.argmax(self(seq)[-1])
            # Add it to the sequence
            seq = jnp.concatenate((seq, idx[None]))

        # Return the sampled sequence
        return seq


class AdamState(NamedTuple):
    """
    This is a named tuple for storing Adam optimizer state for a parameter
    """
    m: jnp.ndarray
    v: jnp.ndarray


class Adam:
    """
    <a id="Adam"></a>

    ## Adam Optimizer

    This is from paper
     [Adam: A Method for Stochastic Optimization](https://papers.labml.ai/paper/1412.6980).

    For parameter $\theta_t$ and gradient $g_t$ at step $t$, the Adam update is,

    \begin{align}
    m_t &\leftarrow \beta_1 m_{t-1} + (1 - \beta_1) \cdot g_t \\
    v_t &\leftarrow \beta_2 v_{t-1} + (1 - \beta_2) \cdot g_t^2 \\
    \hat{m}_t &\leftarrow \frac{m_t}{1-\beta_1^t} \\
    \hat{v}_t &\leftarrow \frac{v_t}{1-\beta_2^t} \\
    \theta_t &\leftarrow \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
    \end{align}

    where $\alpha$, $\beta_1$, $\beta_2$ and $\epsilon$ are scalar hyper parameters.
    $m_t$ and $v_t$ are first and second order moments.
    $\hat{m}_t$  and $\hat{v}_t$ are biased corrected moments.
    $\epsilon$ is used as a fix for division by zero error, but also acts as a form of a hyper-parameter
    that acts against variance in gradients.
    """

    def __init__(self, params: Dict,
                 lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-16, ):
        """
        * `params` is the tree-map of parameters
        * `lr` is the learning rate $\alpha$
        * `betas` is a tuple of ($\beta_1$, $\beta_2$)
        * `eps` is $\hat{\epsilon}$`
        """

        super().__init__()
        self.lr = lr
        self.betas = betas
        self.eps = eps

        # States for each parameter
        self.states = jax.tree.map(self._init_state, params)
        # Optimized step function
        self._step_jit = jax.jit(self._step)
        # Number of steps taken $t$
        self._n_steps = 0
        # Optimized update state function
        self._update_state_jit = jax.jit(self._update_state)

    def _init_state(self, param: jnp.ndarray):
        """
        Initialize the state for a given parameter
        """
        return AdamState(jnp.zeros_like(param), jnp.zeros_like(param))

    def step(self, params: Dict, grads: Dict):
        """
        ## Step function

        * `params` is a tree-map of parameters
        * `grads` is a tree-map of gradients
        """
        # Increment step $t$
        self._n_steps += 1
        # Update states for each parameter
        self.states = jax.tree.map(self._update_state_jit, grads, self.states)
        # Return updated parameters $\theta_t$
        return jax.tree.map(partial(self._step_jit, self._n_steps), params, self.states)

    def _step(self, n_steps: int, param: jnp.ndarray, state: AdamState):
        """
        ### Update parameters

        This performs a Adam update on the given parameter
        """

        # Bias corrections for $\hat{m}_t$: $1 - \beta_1^t$ and for $\hat{v}_t$: $1 - \beta_2^t$
        bias_correction = [1 - beta ** n_steps for beta in self.betas]
        # Uncorrected first and second moments $m_t$ and $v_t$
        m, v = state

        # $\alpha \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$
        step_size = self.lr * (bias_correction[1] ** 0.5) / bias_correction[0]
        # $\sqrt{v_t} + \hat{\epsilon}$
        den = (v ** 0.5) + self.eps

        # $\theta_t \leftarrow \theta_{t-1} - \alpha \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \cdot
        #  \frac{m_t}{\sqrt{v_t} + \hat{\epsilon}}$
        return param - step_size * m / den

    def _update_state(self, grad, state: AdamState):
        """
        ### Update state

        This updates uncorrected first and second moments $m_t$ and $v_t$
        """
        # Uncorrected first and second moments $m_{t-1}$ and $v_{t-1}$
        m, v = state
        # Clip gradients
        grad = jnp.clip(grad, -1, 1)
        # $$m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) \cdot g_t$$
        m = self.betas[0] * m + grad * (1 - self.betas[0])
        # $$v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) \cdot g_t^2$$
        v = self.betas[1] * v + (grad ** 2) * (1 - self.betas[1])

        # Return the new state
        return AdamState(m, v)


class TinyShakespeare:
    """
    <a id="Dataset"></a>

    ## Tiny Shakespeare dataset
    """

    def __init__(self, rnd_key: jax.random.PRNGKey, seq_len: int, batch_size: int):
        """
        * `rnd_key` is the PRNG state
        * `seq_len` is the sequence length of a sample
        * `batch_size` is the batch size
        """

        self.batch_size = batch_size
        # PRNG key for shuffling the samples
        _, self.rnd_key = jax.random.split(rnd_key)

        # Local path of the text file
        path = lab.get_data_path() / 'tiny_shakespeare.txt'
        # Download if it doesn't exist
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        if not path.exists():
            download_file(url, path)

        # Read the file
        with open(str(path), 'r') as f:
            self.text = f.read()

        # Get the characters/tokens
        tokens = sorted(list(set(self.text)))

        # Number of tokens
        self.n_tokens = len(tokens)
        # Map tokens to ids
        self.stoi = {t: i for i, t in enumerate(tokens)}
        # Id to token/character
        self.itos = tokens

        # As a list of ids
        data = jnp.array([self.stoi[s] for s in list(self.text)])
        # Number of batches
        self.n_batches = len(data) // (seq_len * batch_size)
        # Truncate
        data = data[:self.n_batches * seq_len * batch_size]
        # Reshape into a samples (better to use random offsets, but lets ignore that here)
        self.data = data.reshape((-1, seq_len))
        # List of sample indexes
        self.idx = jnp.arange(len(self.data))

    def __iter__(self):
        """
        Setup for iteration
        """
        # Iteration step
        self._iter_idx = 0
        # Split PRNG key
        self.rnd_key, rnd_key = jax.random.split(self.rnd_key)
        # Shuffle sample indexes
        self.idx = jax.random.permutation(rnd_key, self.idx)

        #
        return self

    def __len__(self):
        """
        Number of batches
        """
        return self.n_batches

    def __next__(self):
        """
        Get next batch
        """

        # Stop iteration after iterating through all batches
        if self._iter_idx >= self.n_batches:
            raise StopIteration()

        # Sample indexes for the batch
        idx = self.idx[self._iter_idx * self.batch_size:(self._iter_idx + 1) * self.batch_size]
        # Increment iteration step
        self._iter_idx += 1

        # Return samples
        return self.data[idx]


def main():
    """
    <a id="Experiment"></a>

    ## Run the experiment
    """

    # Create experiment
    experiment.create(name='jax')
    # Create PRNG key
    rnd_key = jax.random.PRNGKey(0)
    # Create dataset
    dataset = TinyShakespeare(rnd_key, seq_len=32, batch_size=128)

    # Create the model
    model = AutoregressiveTransformer(rnd_key, dataset.n_tokens,
                                      d_model=128, n_layers=3, heads=8, d_ff=512)
    # Get model parameters
    params = model.get_params()

    # JAX compiled pure sampling function
    pure_sample_fn = jax.jit(model.purify(model.sample))
    # JAX compiled pure function to get logits for a batch.
    # First we transform `model.__call__` to a pure function which accepts two arguments:
    # parameters, and input sequence.
    # Next we vectorize the function to process a batch of samples. `in_axes` specifies which arguments
    # to parallelize and along which axis. `(None, 0)` means we have the same parameters but parallelize
    # the inputs across the first axis.
    # `out_axes` specifies along which axis to merge the results.
    pure_forward_fn = jax.jit(jax.vmap(model.purify(model.__call__),
                                       in_axes=(None, 0), out_axes=0))
    # Similarly we vectorize loss computation
    pure_loss_fn = jax.jit(jax.vmap(model.purify(model.get_loss),
                                    in_axes=(None, 0), out_axes=0))

    # A function to get mean loss
    def get_loss(params, seq):
        return pure_loss_fn(params, seq).mean()

    # A function to compute gradients for the first argument (parameters)
    grad_loss_fn = jax.jit(jax.grad(get_loss, argnums=0))

    # Create optimizer
    optimizer = Adam(params)

    # Start the experiment
    with experiment.start():
        # Iterate for 32 epochs
        for epoch in monit.loop(32):
            # Iterate through batches
            for data in monit.iterate('Train', dataset):
                # Compute and log the loss
                loss = get_loss(params, data)
                tracker.save('loss', np.asarray(loss))
                # Get the gradients
                grads = grad_loss_fn(params, data)
                # Update parameters
                params = optimizer.step(params, grads)

            #
            tracker.new_line()
            # Log a sample after each epoch
            prompt = [dataset.stoi[c] for c in 'It ']
            sampled = pure_sample_fn(params, jnp.array(prompt))[len(prompt):]
            sampled = ''.join([dataset.itos[i] for i in sampled])
            sampled = sampled.replace('\n', '\\n')
            logger.log(('It ', Text.meta), (sampled, Text.value))


#
if __name__ == '__main__':
    main()