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

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/a44333ea251411ec8007d1a1762ed686)
"""


import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from labml_nn.diffusion.ddpm.utils import gather


class GaussianDiffusion:
    def __init__(self, eps_model: nn.Module, n_steps, device):
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
