"""
---
title: Denoising Diffusion Probabilistic Models (DDPM)
summary: >
  PyTorch implementation and tutorial of the paper
  Denoising Diffusion Probabilistic Models (DDPM).
---

# Denoising Diffusion Probabilistic Models (DDPM)

This is a [PyTorch](https://pytorch.org) implementation/tutorial of the paper
[Denoising Diffusion Probabilistic Models](https://papers.labml.ai/paper/2006.11239).

## Forward Process

The forward process adds noise to the data $x_0 \sim q(x_0)$, for $T$ timesteps.

\begin{align}
q(x_t | x_{t-1}) = \mathcal{N}\big(x_t; \sqrt{1-  \beta_t} x_{t-1}, \beta_t \mathbf{I}\big) \\
q(x_{1:T} | x_0) = \prod_{t = 1}^{T} q(x_t | x_{t-1})
\end{align}

where $\beta_1, \dots, \beta_T$ is the variance schedule.

We can sample $x_t$ at any timestep $t$ with,

\begin{align}
q(x_t|x_0) &= \mathcal{N}(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I}) \\
\end{align}

where $$\alpha_t = 1 - \beta_t \text{ and } \bar\alpha_t = \prod_{s=1}^t \alpha_s$$

The forward process posterior conditioned by $x_0$ is,

\begin{align}
q(x_{t-1}|x_t, x_0) &= \mathcal{N}(x_{t-1}; \tilde\mu_t(x_t, x_0), \tilde\beta_t \mathbf{I}) \\
\tilde\mu_t(x_t, x_0) &= \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
                         + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1}}{1-\bar\alpha_t}x_t \\
\tilde\beta_t &= \frac{1 - \bar\alpha_{t-1}}{a}
\end{align}

## Reverse Process

The reverse process removes noise starting at $p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
for $T$ time steps.

\begin{align}
p_\theta(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \beta_t, \Sigma_\theta(x_t, t)\big) \\
p_\theta(x_{0:T}) &= p_\theta(x_T) \prod_{t = 1}^{T} p_\theta(x_{t-1} | x_t) \\
p_\theta(x_0) &= \int p_\theta(x_{0:T}) dx_{1:T}
\end{align}

$\theta$ are the parameters we train.

## Loss

We optimize the ELBO (from Jenson's inequality) on the negative log likelihood.

\begin{align}
\mathbb{E}[-\log p_\theta(x_0)]
 &\le \mathbb{E}_q [ -\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} ] \\
 &=L
\end{align}

The loss can be rewritten as  follows.

\begin{align}
L
 &= \mathbb{E}_q [ -\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} ] \\
 &= \mathbb{E}_q [ -\log p(x_T) - \sum_{t=1}^T \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})} ] \\
 &= \mathbb{E}_q [
  -\log \frac{p(x_T)}{q(x_T|x_0)}
  -\sum_{t=2}^T \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}
  -\log p_\theta(x_0|x_1)] \\
 &= \mathbb{E}_q [
   D_{KL}(q(x_T|x_0) \Vert p(x_T))
  +\sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t,x_0) \Vert p_\theta(x_{t-1}|x_t))
  -\log p_\theta(x_0|x_1)]
\end{align}

$D_{KL}(q(x_T|x_0) \Vert p(x_T))$ is constant since we keep $\beta_1, \dots, \beta_T$ constant.

### Computing $L_{t-1} = D_{KL}(q(x_{t-1}|x_t,x_0) \Vert p_\theta(x_{t-1}|x_t))$

The paper sets $\Sigma_\theta(x_t, t) = \sigma_t^2 \mathbf{I}$ where $\sigma_t^2$ is set to constants
$\beta_t$ or $\tilde\beta_t$.

Then,
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 \mathbf{I} \big)$$

For given noise $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ using $q(x_t|x_0)$

\begin{align}
x_t(x_0, \epsilon) &= \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon \\
x_0 &= \frac{1}{\sqrt{\bar\alpha_t}} \Big(x_t(x_0, \epsilon) -  \sqrt{1-\bar\alpha_t}\epsilon\Big)
\end{align}

This gives,

\begin{align}
L_{t-1}
 &= D_{KL}(q(x_{t-1}|x_t,x_0) \Vert p_\theta(x_{t-1}|x_t)) \\
 &= \mathbb{E}_q \Bigg[ \frac{1}{2\sigma_t^2} \Vert \tilde\mu(x_t, x_0) - \mu_theta(x_t, t) \Vert^2 \Bigg] \\
 &= \mathbb{E}_{x_0, \epsilon} \Bigg[ \frac{1}{2\sigma_t^2}
  \bigg\Vert \frac{1}{\sqrt{\alpha_t}} \Big(
  x_t(x_0, \epsilon) - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}} \epsilon
  \Big) - \mu_\theta(x_t(x_0, \epsilon), t) \bigg\Vert^2 \Bigg] \\
\end{align}

Re-parameterizing

\begin{align}
\mu_\theta(x_t, t) &= \tilde\mu \bigg(x_t,
  \frac{1}{\sqrt{\bar\alpha_t}} \Big(x_t -  \sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t, t) \Big) \bigg) \\
  &= \frac{1}{\sqrt{\bar\alpha_t}} \Big(x_t -  \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t) \Big)
\end{align}

where $\epsilon_theta$ is a learned function that predicts $\epsilon$ given $(x_t, t)$.

This gives,

\begin{align}
L_{t-1}
&= \mathbb{E}_{x_0, \epsilon} \Bigg[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar\alpha_t)}
  \Big\Vert
  \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
  \Big\Vert^2 \Bigg]
\end{align}

That is, we are training to predict the noise.

### Simplified loss

$$L_simple(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
\bigg\Vert^2 \Bigg]$$

This minimizes $-\log p_\theta(x_0|x_1)$ when $t=1$ and $L_{t-1}$ for $t\gt1$ discarding the
weighting in $L_{t-1}$. Discarding the weights $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar\alpha_t)}$
increase the weight given to higher $t$ (which have higher noise levels), therefore increasing the sample quality.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/a44333ea251411ec8007d1a1762ed686)
"""


import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from labml_nn.diffusion.ddpm.utils import gather


class DenoiseDiffusion:
    """
    ## Denoise Diffusion
    """
    def __init__(self, eps_model: nn.Module, n_steps, device):
        """
        * `eps_model` is $\eps_\theta(x_t, t)$ model
        * `n_steps` is $t$
        """
        super().__init__()
        self.eps_model = eps_model

        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)

        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.n_steps = n_steps

        self.sigma = self.beta

    def q_xt_x0(self, x0, t):
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * noise

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.eps_model(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma, t) ** .5

        noise = torch.randn(xt.shape, device=xt.device)
        return mean + var * noise

    def loss(self, x0, noise=None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, noise=noise)
        eps_theta = self.eps_model(xt, t)

        return F.mse_loss(noise, eps_theta)
