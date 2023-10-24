"""
---
title: Denoising Diffusion Implicit Models (DDIM) Sampling
summary: >
 Annotated PyTorch implementation/tutorial of
 Denoising Diffusion Implicit Models (DDIM) Sampling
 for stable diffusion model.
---

# Denoising Diffusion Implicit Models (DDIM) Sampling

This implements DDIM sampling from the paper
[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
"""

from typing import Optional, List

import numpy as np
import torch

from labml import monit
from labml_nn.diffusion.stable_diffusion.latent_diffusion import LatentDiffusion
from labml_nn.diffusion.stable_diffusion.sampler import DiffusionSampler


class DDIMSampler(DiffusionSampler):
    """
    ## DDIM Sampler

    This extends the [`DiffusionSampler` base class](index.html).

    DDIM samples images by repeatedly removing noise by sampling step by step using,

    \begin{align}
    x_{\tau_{i-1}} &= \sqrt{\alpha_{\tau_{i-1}}}\Bigg(
            \frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}
            \Bigg) \\
            &+ \sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i}) \\
            &+ \sigma_{\tau_i} \epsilon_{\tau_i}
    \end{align}

    where $\epsilon_{\tau_i}$ is random noise,
    $\tau$ is a subsequence of $[1,2,\dots,T]$ of length $S$,
    and
    $$\sigma_{\tau_i} =
    \eta \sqrt{\frac{1 - \alpha_{\tau_{i-1}}}{1 - \alpha_{\tau_i}}}
    \sqrt{1 - \frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}$$

    Note that, $\alpha_t$ in DDIM paper refers to ${\color{lightgreen}\bar\alpha_t}$ from [DDPM](ddpm.html).
    """

    model: LatentDiffusion

    def __init__(self, model: LatentDiffusion, n_steps: int, ddim_discretize: str = "uniform", ddim_eta: float = 0.):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        :param n_steps: is the number of DDIM sampling steps, $S$
        :param ddim_discretize: specifies how to extract $\tau$ from $[1,2,\dots,T]$.
            It can be either `uniform` or `quad`.
        :param ddim_eta: is $\eta$ used to calculate $\sigma_{\tau_i}$. $\eta = 0$ makes the
            sampling process deterministic.
        """
        super().__init__(model)
        # Number of steps, $T$
        self.n_steps = model.n_steps

        # Calculate $\tau$ to be uniformly distributed across $[1,2,\dots,T]$
        if ddim_discretize == 'uniform':
            c = self.n_steps // n_steps
            self.time_steps = np.asarray(list(range(0, self.n_steps, c))) + 1
        # Calculate $\tau$ to be quadratically distributed across $[1,2,\dots,T]$
        elif ddim_discretize == 'quad':
            self.time_steps = ((np.linspace(0, np.sqrt(self.n_steps * .8), n_steps)) ** 2).astype(int) + 1
        else:
            raise NotImplementedError(ddim_discretize)

        with torch.no_grad():
            # Get ${\color{lightgreen}\bar\alpha_t}$
            alpha_bar = self.model.alpha_bar

            # $\alpha_{\tau_i}$
            self.ddim_alpha = alpha_bar[self.time_steps].clone().to(torch.float32)
            # $\sqrt{\alpha_{\tau_i}}$
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            # $\alpha_{\tau_{i-1}}$
            self.ddim_alpha_prev = torch.cat([alpha_bar[0:1], alpha_bar[self.time_steps[:-1]]])

            # $$\sigma_{\tau_i} =
            # \eta \sqrt{\frac{1 - \alpha_{\tau_{i-1}}}{1 - \alpha_{\tau_i}}}
            # \sqrt{1 - \frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}$$
            self.ddim_sigma = (ddim_eta *
                               ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                                (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** .5)

            # $\sqrt{1 - \alpha_{\tau_i}}$
            self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5

    @torch.no_grad()
    def sample(self,
               shape: List[int],
               cond: torch.Tensor,
               repeat_noise: bool = False,
               temperature: float = 1.,
               x_last: Optional[torch.Tensor] = None,
               uncond_scale: float = 1.,
               uncond_cond: Optional[torch.Tensor] = None,
               skip_steps: int = 0,
               ):
        """
        ### Sampling Loop

        :param shape: is the shape of the generated images in the
            form `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_{\tau_S}$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        :param skip_steps: is the number of time steps to skip $i'$. We start sampling from $S - i'$.
            And `x_last` is then $x_{\tau_{S - i'}}$.
        """

        # Get device and batch size
        device = self.model.device
        bs = shape[0]

        # Get $x_{\tau_S}$
        x = x_last if x_last is not None else torch.randn(shape, device=device)

        # Time steps to sample at $\tau_{S - i'}, \tau_{S - i' - 1}, \dots, \tau_1$
        time_steps = np.flip(self.time_steps)[skip_steps:]

        for i, step in monit.enum('Sample', time_steps):
            # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
            index = len(time_steps) - i - 1
            # Time step $\tau_i$
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample $x_{\tau_{i-1}}$
            x, pred_x0, e_t = self.p_sample(x, cond, ts, step, index=index,
                                            repeat_noise=repeat_noise,
                                            temperature=temperature,
                                            uncond_scale=uncond_scale,
                                            uncond_cond=uncond_cond)

        # Return $x_0$
        return x

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, step: int, index: int, *,
                 repeat_noise: bool = False,
                 temperature: float = 1.,
                 uncond_scale: float = 1.,
                 uncond_cond: Optional[torch.Tensor] = None):
        """
        ### Sample $x_{\tau_{i-1}}$

        :param x: is $x_{\tau_i}$ of shape `[batch_size, channels, height, width]`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param t: is $\tau_i$ of shape `[batch_size]`
        :param step: is the step $\tau_i$ as an integer
        :param index: is index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
        :param repeat_noise: specified whether the noise should be same for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """

        # Get $\epsilon_\theta(x_{\tau_i})$
        e_t = self.get_eps(x, t, c,
                           uncond_scale=uncond_scale,
                           uncond_cond=uncond_cond)

        # Calculate $x_{\tau_{i - 1}}$ and predicted $x_0$
        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(e_t, index, x,
                                                      temperature=temperature,
                                                      repeat_noise=repeat_noise)

        #
        return x_prev, pred_x0, e_t

    def get_x_prev_and_pred_x0(self, e_t: torch.Tensor, index: int, x: torch.Tensor, *,
                               temperature: float,
                               repeat_noise: bool):
        """
        ### Sample $x_{\tau_{i-1}}$ given $\epsilon_\theta(x_{\tau_i})$
        """

        # $\alpha_{\tau_i}$
        alpha = self.ddim_alpha[index]
        # $\alpha_{\tau_{i-1}}$
        alpha_prev = self.ddim_alpha_prev[index]
        # $\sigma_{\tau_i}$
        sigma = self.ddim_sigma[index]
        # $\sqrt{1 - \alpha_{\tau_i}}$
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]

        # Current prediction for $x_0$,
        # $$\frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}$$
        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)
        # Direction pointing to $x_t$
        # $$\sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i})$$
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t

        # No noise is added, when $\eta = 0$
        if sigma == 0.:
            noise = 0.
        # If same noise is used for all samples in the batch
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
            # Different noise for each sample
        else:
            noise = torch.randn(x.shape, device=x.device)

        # Multiply noise by the temperature
        noise = noise * temperature

        #  \begin{align}
        #     x_{\tau_{i-1}} &= \sqrt{\alpha_{\tau_{i-1}}}\Bigg(
        #             \frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}
        #             \Bigg) \\
        #             &+ \sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i}) \\
        #             &+ \sigma_{\tau_i} \epsilon_{\tau_i}
        #  \end{align}
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise

        #
        return x_prev, pred_x0

    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        """
        ### Sample from $q_{\sigma,\tau}(x_{\tau_i}|x_0)$

        $$q_{\sigma,\tau}(x_t|x_0) =
         \mathcal{N} \Big(x_t; \sqrt{\alpha_{\tau_i}} x_0, (1-\alpha_{\tau_i}) \mathbf{I} \Big)$$

        :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        :param index: is the time step $\tau_i$ index $i$
        :param noise: is the noise, $\epsilon$
        """

        # Random noise, if noise is not specified
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample from
        #  $$q_{\sigma,\tau}(x_t|x_0) =
        #          \mathcal{N} \Big(x_t; \sqrt{\alpha_{\tau_i}} x_0, (1-\alpha_{\tau_i}) \mathbf{I} \Big)$$
        return self.ddim_alpha_sqrt[index] * x0 + self.ddim_sqrt_one_minus_alpha[index] * noise

    @torch.no_grad()
    def paint(self, x: torch.Tensor, cond: torch.Tensor, t_start: int, *,
              orig: Optional[torch.Tensor] = None,
              mask: Optional[torch.Tensor] = None, orig_noise: Optional[torch.Tensor] = None,
              uncond_scale: float = 1.,
              uncond_cond: Optional[torch.Tensor] = None,
              ):
        """
        ### Painting Loop

        :param x: is $x_{S'}$ of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param t_start: is the sampling step to start from, $S'$
        :param orig: is the original image in latent page which we are in paining.
            If this is not provided, it'll be an image to image transformation.
        :param mask: is the mask to keep the original image.
        :param orig_noise: is fixed noise to be added to the original image.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        # Get  batch size
        bs = x.shape[0]

        # Time steps to sample at $\tau_{S`}, \tau_{S' - 1}, \dots, \tau_1$
        time_steps = np.flip(self.time_steps[:t_start])

        for i, step in monit.enum('Paint', time_steps):
            # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
            index = len(time_steps) - i - 1
            # Time step $\tau_i$
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample $x_{\tau_{i-1}}$
            x, _, _ = self.p_sample(x, cond, ts, step, index=index,
                                    uncond_scale=uncond_scale,
                                    uncond_cond=uncond_cond)

            # Replace the masked area with original image
            if orig is not None:
                # Get the $q_{\sigma,\tau}(x_{\tau_i}|x_0)$ for original image in latent space
                orig_t = self.q_sample(orig, index, noise=orig_noise)
                # Replace the masked area
                x = orig_t * mask + x * (1 - mask)

        #
        return x
