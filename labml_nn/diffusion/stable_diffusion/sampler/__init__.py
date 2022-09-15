"""
---
title: Sampling algorithms for stable diffusion
summary: >
 Annotated PyTorch implementation/tutorial of
 sampling algorithms
 for stable diffusion model.
---

# Sampling algorithms for [stable diffusion](../index.html)

We have implemented the following [sampling algorithms](sampler/index.html):

* [Denoising Diffusion Probabilistic Models (DDPM) Sampling](ddpm.html)
* [Denoising Diffusion Implicit Models (DDIM) Sampling](ddim.html)
"""

from typing import Optional, List

import torch

from labml_nn.diffusion.stable_diffusion.latent_diffusion import LatentDiffusion


class DiffusionSampler:
    """
    ## Base class for sampling algorithms
    """
    model: LatentDiffusion

    def __init__(self, model: LatentDiffusion):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """
        super().__init__()
        # Set the model $\epsilon_\text{cond}(x_t, c)$
        self.model = model
        # Get number of steps the model was trained with $T$
        self.n_steps = model.n_steps

    def get_eps(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, *,
                uncond_scale: float, uncond_cond: Optional[torch.Tensor]):
        """
        ## Get $\epsilon(x_t, c)$

        :param x: is $x_t$ of shape `[batch_size, channels, height, width]`
        :param t: is $t$ of shape `[batch_size]`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        # When the scale $s = 1$
        # $$\epsilon_\theta(x_t, c) = \epsilon_\text{cond}(x_t, c)$$
        if uncond_cond is None or uncond_scale == 1.:
            return self.model(x, t, c)

        # Duplicate $x_t$ and $t$
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        # Concatenated $c$ and $c_u$
        c_in = torch.cat([uncond_cond, c])
        # Get $\epsilon_\text{cond}(x_t, c)$ and $\epsilon_\text{cond}(x_t, c_u)$
        e_t_uncond, e_t_cond = self.model(x_in, t_in, c_in).chunk(2)
        # Calculate
        # $$\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$$
        e_t = e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)

        #
        return e_t

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
        :param x_last: is $x_T$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        :param skip_steps: is the number of time steps to skip.
        """
        raise NotImplementedError()

    def paint(self, x: torch.Tensor, cond: torch.Tensor, t_start: int, *,
              orig: Optional[torch.Tensor] = None,
              mask: Optional[torch.Tensor] = None, orig_noise: Optional[torch.Tensor] = None,
              uncond_scale: float = 1.,
              uncond_cond: Optional[torch.Tensor] = None,
              ):
        """
        ### Painting Loop

        :param x: is $x_{T'}$ of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param t_start: is the sampling step to start from, $T'$
        :param orig: is the original image in latent page which we are in paining.
        :param mask: is the mask to keep the original image.
        :param orig_noise: is fixed noise to be added to the original image.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        raise NotImplementedError()

    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        """
        ### Sample from $q(x_t|x_0)$

        :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        :param index: is the time step $t$ index
        :param noise: is the noise, $\epsilon$
        """
        raise NotImplementedError()
