"""
---
title: HyperNetworks - HyperLSTM
summary: A PyTorch implementation/tutorial of HyperLSTM introduced in paper HyperNetworks.
---

# HyperNetworks - HyperLSTM

We have implemented HyperLSTM introduced in paper
[HyperNetworks](https://arxiv.org/abs/1609.09106), with annotations
using [PyTorch](https://pytorch.org).
[This blog post](https://blog.otoro.net/2016/09/28/hyper-networks/)
by David Ha gives a good explanation of HyperNetworks.

We have an experiment that trains a HyperLSTM to predict text on Shakespeare dataset.
Here's the link to code: [`experiment.py`](experiment.html)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/hypernetworks/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/9e7f39e047e811ebbaff2b26e3148b3d)

HyperNetworks use a smaller network to generate weights of a larger network.
There are two variants: static hyper-networks and dynamic hyper-networks.
Static HyperNetworks have smaller networks that generate weights (kernels)
of a convolutional network. Dynamic HyperNetworks generate parameters of a
recurrent neural network
for each step. This is an implementation of the latter.

## Dynamic HyperNetworks
In a RNN the parameters stay constant for each step.
Dynamic HyperNetworks generate different parameters for each step.
HyperLSTM has the structure of a LSTM but the parameters of
each step are changed by a smaller LSTM network.

In the basic form, a Dynamic HyperNetwork has a smaller recurrent network that generates
a feature vector corresponding to each parameter tensor of the larger recurrent network.
Let's say the larger network has some parameter $\color{cyan}{W_h}$ the smaller network generates a feature
vector $z_h$ and we dynamically compute $\color{cyan}{W_h}$ as a linear transformation of $z_h$.
For instance $\color{cyan}{W_h} =  \langle W_{hz}, z_h \rangle$ where
$W_{hz}$ is a 3-d tensor parameter and $\langle . \rangle$ is a tensor-vector multiplication.
$z_h$ is usually a linear transformation of the output of the smaller recurrent network.

### Weight scaling instead of computing

Large recurrent networks have large dynamically computed parameters.
These are calculated using linear transformation of feature vector $z$.
And this transformation requires an even larger weight tensor.
That is, when $\color{cyan}{W_h}$ has shape $N_h \times N_h$,
$W_{hz}$ will be $N_h \times N_h \times N_z$.

To overcome this, we compute the weight parameters of the recurrent network by
dynamically scaling each row of a matrix of same size.
\begin{align}
d(z) = W_{hz} z_h \\
\\
\color{cyan}{W_h} =
\begin{pmatrix}
d_0(z) W_{hd_0} \\
d_1(z) W_{hd_1} \\
... \\
d_{N_h}(z) W_{hd_{N_h}} \\
\end{pmatrix}
\end{align}
where $W_{hd}$ is a $N_h \times N_h$ parameter matrix.

We can further optimize this when we compute $\color{cyan}{W_h} h$,
as
$$\color{lightgreen}{d(z) \odot (W_{hd} h)}$$
where $\odot$ stands for element-wise multiplication.
"""

from typing import Optional, Tuple

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.lstm import LSTMCell


class HyperLSTMCell(Module):
    """
    ## HyperLSTM Cell

    For HyperLSTM the smaller network and the larger network both have the LSTM structure.
    This is defined in Appendix A.2.2 in the paper.
    """

    def __init__(self, input_size: int, hidden_size: int, hyper_size: int, n_z: int):
        """
        `input_size` is the size of the input $x_t$,
        `hidden_size` is the size of the LSTM, and
        `hyper_size` is the size of the smaller LSTM that alters the weights of the larger outer LSTM.
        `n_z` is the size of the feature vectors used to alter the LSTM weights.

        We use the output of the smaller LSTM to compute $z_h^{i,f,g,o}$, $z_x^{i,f,g,o}$ and
        $z_b^{i,f,g,o}$ using linear transformations.
        We calculate $d_h^{i,f,g,o}(z_h^{i,f,g,o})$, $d_x^{i,f,g,o}(z_x^{i,f,g,o})$, and
        $d_b^{i,f,g,o}(z_b^{i,f,g,o})$ from these, using linear transformations again.
        These are then used to scale the rows of weight and bias tensors of the main LSTM.

        📝 Since the computation of $z$ and $d$ are two sequential linear transformations
        these can be combined into a single linear transformation.
        However we've implemented this separately so that it matches with the description
        in the paper.
        """
        super().__init__()

        # The input to the hyperLSTM is
        # $$
        # \hat{x}_t = \begin{pmatrix}
        # h_{t-1} \\
        # x_t
        # \end{pmatrix}
        # $$
        # where $x_t$ is the input and $h_{t-1}$ is the output of the outer LSTM at previous step.
        # So the input size is `hidden_size + input_size`.
        #
        # The output of hyperLSTM is $\hat{h}_t$ and $\hat{c}_t$.
        self.hyper = LSTMCell(hidden_size + input_size, hyper_size, layer_norm=True)

        # $$z_h^{i,f,g,o} = lin_{h}^{i,f,g,o}(\hat{h}_t)$$
        # 🤔 In the paper it was specified as
        # $$z_h^{i,f,g,o} = lin_{h}^{i,f,g,o}(\hat{h}_{\color{red}{t-1}})$$
        # I feel that it's a typo.
        self.z_h = nn.Linear(hyper_size, 4 * n_z)
        # $$z_x^{i,f,g,o} = lin_x^{i,f,g,o}(\hat{h}_t)$$
        self.z_x = nn.Linear(hyper_size, 4 * n_z)
        # $$z_b^{i,f,g,o} = lin_b^{i,f,g,o}(\hat{h}_t)$$
        self.z_b = nn.Linear(hyper_size, 4 * n_z, bias=False)

        # $$d_h^{i,f,g,o}(z_h^{i,f,g,o}) = lin_{dh}^{i,f,g,o}(z_h^{i,f,g,o})$$
        d_h = [nn.Linear(n_z, hidden_size, bias=False) for _ in range(4)]
        self.d_h = nn.ModuleList(d_h)
        # $$d_x^{i,f,g,o}(z_x^{i,f,g,o}) = lin_{dx}^{i,f,g,o}(z_x^{i,f,g,o})$$
        d_x = [nn.Linear(n_z, hidden_size, bias=False) for _ in range(4)]
        self.d_x = nn.ModuleList(d_x)
        # $$d_b^{i,f,g,o}(z_b^{i,f,g,o}) = lin_{db}^{i,f,g,o}(z_b^{i,f,g,o})$$
        d_b = [nn.Linear(n_z, hidden_size) for _ in range(4)]
        self.d_b = nn.ModuleList(d_b)

        # The weight matrices $W_h^{i,f,g,o}$
        self.w_h = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, hidden_size)) for _ in range(4)])
        # The weight matrices $W_x^{i,f,g,o}$
        self.w_x = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, input_size)) for _ in range(4)])

        # Layer normalization
        self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
        self.layer_norm_c = nn.LayerNorm(hidden_size)

    def __call__(self, x: torch.Tensor,
                 h: torch.Tensor, c: torch.Tensor,
                 h_hat: torch.Tensor, c_hat: torch.Tensor):
        # $$
        # \hat{x}_t = \begin{pmatrix}
        # h_{t-1} \\
        # x_t
        # \end{pmatrix}
        # $$
        x_hat = torch.cat((h, x), dim=-1)
        # $$\hat{h}_t, \hat{c}_t = lstm(\hat{x}_t, \hat{h}_{t-1}, \hat{c}_{t-1})$$
        h_hat, c_hat = self.hyper(x_hat, h_hat, c_hat)

        # $$z_h^{i,f,g,o} = lin_{h}^{i,f,g,o}(\hat{h}_t)$$
        z_h = self.z_h(h_hat).chunk(4, dim=-1)
        # $$z_x^{i,f,g,o} = lin_x^{i,f,g,o}(\hat{h}_t)$$
        z_x = self.z_x(h_hat).chunk(4, dim=-1)
        # $$z_b^{i,f,g,o} = lin_b^{i,f,g,o}(\hat{h}_t)$$
        z_b = self.z_b(h_hat).chunk(4, dim=-1)

        # We calculate $i$, $f$, $g$ and $o$ in a loop
        ifgo = []
        for i in range(4):
            # $$d_h^{i,f,g,o}(z_h^{i,f,g,o}) = lin_{dh}^{i,f,g,o}(z_h^{i,f,g,o})$$
            d_h = self.d_h[i](z_h[i])
            # $$d_x^{i,f,g,o}(z_x^{i,f,g,o}) = lin_{dx}^{i,f,g,o}(z_x^{i,f,g,o})$$
            d_x = self.d_x[i](z_x[i])

            # \begin{align}
            # {i,f,g,o} = LN(&\color{lightgreen}{d_h^{i,f,g,o}(z_h) \odot (W_h^{i,f,g,o} h_{t-1})} \\
            #              + &\color{lightgreen}{d_x^{i,f,g,o}(z_x) \odot (W_h^{i,f,g,o} x_t)} \\
            #              + &d_b^{i,f,g,o}(z_b))
            # \end{align}
            y = d_h * torch.einsum('ij,bj->bi', self.w_h[i], h) + \
                d_x * torch.einsum('ij,bj->bi', self.w_x[i], x) + \
                self.d_b[i](z_b[i])

            ifgo.append(self.layer_norm[i](y))

        # $$i_t, f_t, g_t, o_t$$
        i, f, g, o = ifgo

        # $$c_t = \sigma(f_t) \odot c_{t-1} + \sigma(i_t) \odot \tanh(g_t) $$
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)

        # $$h_t = \sigma(o_t) \odot \tanh(LN(c_t))$$
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))

        return h_next, c_next, h_hat, c_hat


class HyperLSTM(Module):
    """
    # HyperLSTM module
    """
    def __init__(self, input_size: int, hidden_size: int, hyper_size: int, n_z: int, n_layers: int):
        """
        Create a network of `n_layers` of HyperLSTM.
        """

        super().__init__()

        # Store sizes to initialize state
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.hyper_size = hyper_size

        # Create cells for each layer. Note that only the first layer gets the input directly.
        # Rest of the layers get the input from the layer below
        self.cells = nn.ModuleList([HyperLSTMCell(input_size, hidden_size, hyper_size, n_z)] +
                                   [HyperLSTMCell(hidden_size, hidden_size, hyper_size, n_z) for _ in
                                    range(n_layers - 1)])

    def __call__(self, x: torch.Tensor,
                 state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        """
        * `x` has shape `[n_steps, batch_size, input_size]` and
        * `state` is a tuple of $h, c, \hat{h}, \hat{c}$.
         $h, c$ have shape `[batch_size, hidden_size]` and
         $\hat{h}, \hat{c}$ have shape `[batch_size, hyper_size]`.
        """
        n_steps, batch_size = x.shape[:2]

        # Initialize the state with zeros if `None`
        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            h_hat = [x.new_zeros(batch_size, self.hyper_size) for _ in range(self.n_layers)]
            c_hat = [x.new_zeros(batch_size, self.hyper_size) for _ in range(self.n_layers)]
        #
        else:
            (h, c, h_hat, c_hat) = state
            # Reverse stack the tensors to get the states of each layer
            #
            # 📝 You can just work with the tensor itself but this is easier to debug
            h, c = list(torch.unbind(h)), list(torch.unbind(c))
            h_hat, c_hat = list(torch.unbind(h_hat)), list(torch.unbind(c_hat))

        # Collect the outputs of the final layer at each step
        out = []
        for t in range(n_steps):
            # Input to the first layer is the input itself
            inp = x[t]
            # Loop through the layers
            for layer in range(self.n_layers):
                # Get the state of the layer
                h[layer], c[layer], h_hat[layer], c_hat[layer] = \
                    self.cells[layer](inp, h[layer], c[layer], h_hat[layer], c_hat[layer])
                # Input to the next layer is the state of this layer
                inp = h[layer]
            # Collect the output $h$ of the final layer
            out.append(h[-1])

        # Stack the outputs and states
        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)
        h_hat = torch.stack(h_hat)
        c_hat = torch.stack(c_hat)

        #
        return out, (h, c, h_hat, c_hat)
