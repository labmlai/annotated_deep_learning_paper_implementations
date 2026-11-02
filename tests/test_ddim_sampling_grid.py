"""Sampling indices and DDIM transitions must describe the same Gaussian path."""

import math

import numpy as np
import pytest
import torch

from labml_nn.diffusion.stable_diffusion.latent_diffusion import LatentDiffusion
from labml_nn.diffusion.stable_diffusion.sampler.ddim import DDIMSampler


class ConstantNoise(torch.nn.Module):
    """Analytically solvable denoiser for a fixed-noise Gaussian trajectory."""
    def __init__(self):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.tensor([[[[.2, -.3], [.4, -.1]]]]))
        self.seen_steps = []

    def forward(self, x, time_steps, context):
        self.seen_steps.extend(time_steps.detach().cpu().tolist())
        return self.noise.expand_as(x)


def diffusion_model(training_steps):
    # Exercise the genuine LatentDiffusion/DiffusionWrapper path. Text/image
    # encoders are unused because sampling receives conditioning/latents directly.
    return LatentDiffusion(unet_model=ConstantNoise(), autoencoder=None, clip_embedder=None,
                           latent_scaling_factor=1., n_steps=training_steps,
                           linear_start=.001, linear_end=.04)


def reference_steps(training_steps, sampling_steps, discretization):
    if sampling_steps == 1:
        return [training_steps - 1]
    if discretization == "uniform":
        return [math.floor((training_steps - 1) * index / (sampling_steps - 1))
                for index in range(sampling_steps)]
    return [math.floor(.8 * training_steps * (index / (sampling_steps - 1)) ** 2 + 1e-12)
            for index in range(sampling_steps)]


@pytest.mark.parametrize("discretization", ["uniform", "quad"])
@pytest.mark.parametrize("sizes", [(8, 8), (10, 3), (1000, 333), (1000, 128), (8, 1), (1000, 50), (1000, 1000)])
def test_sampling_grid_has_requested_count_and_valid_training_indices(discretization, sizes):
    training_steps, sampling_steps = sizes
    model = diffusion_model(training_steps)
    sampler = DDIMSampler(model, sampling_steps, ddim_discretize=discretization)
    assert len(sampler.time_steps) == sampling_steps
    assert np.all(sampler.time_steps >= 0)
    assert np.all(sampler.time_steps < training_steps)
    if discretization == "uniform":
        assert len(np.unique(sampler.time_steps)) == sampling_steps
    # Floating-point quadratic rounding can differ at exact integer boundaries;
    # grid validity/count, not a rounding-direction convention, is its contract.
    if discretization == "uniform" or sampling_steps == 1:
        np.testing.assert_array_equal(sampler.time_steps, reference_steps(*sizes, discretization))
    expected_alpha = model.alpha_bar[sampler.time_steps]
    torch.testing.assert_close(sampler.ddim_alpha, expected_alpha, rtol=0., atol=0.)
    assert sampler.ddim_alpha_prev[0] == 1.
    torch.testing.assert_close(sampler.ddim_alpha_prev[1:], expected_alpha[:-1], rtol=0., atol=0.)


@pytest.mark.parametrize("discretization", ["uniform", "quad"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("eta", [0., .5, 1.])
@pytest.mark.parametrize("index", [0, 3])
def test_transition_matches_ddim_gaussian_formula(discretization, dtype, eta, index):
    model = diffusion_model(12)
    sampler = DDIMSampler(model, 4, ddim_discretize=discretization, ddim_eta=eta)
    clean = torch.tensor([[[[.5, -.8], [1.1, .3]]], [[[.2, .9], [-.7, 1.2]]]], dtype=dtype)
    epsilon = model.model.diffusion_model.noise.detach().to(dtype).expand_as(clean)
    alpha = float(model.alpha_bar[int(sampler.time_steps[index])])
    previous = 1. if index == 0 else float(model.alpha_bar[int(sampler.time_steps[index - 1])])
    current = math.sqrt(alpha) * clean + math.sqrt(1. - alpha) * epsilon
    sigma = eta * math.sqrt((1. - previous) / (1. - alpha) * (1. - alpha / previous))
    torch.manual_seed(43)
    noise = torch.randn(current.shape) if sigma != 0. else 0.
    expected = (math.sqrt(previous) * clean
                + math.sqrt(max(0., 1. - previous - sigma ** 2)) * epsilon + sigma * noise)
    torch.manual_seed(43)
    actual, predicted_clean = sampler.get_x_prev_and_pred_x0(
        epsilon, index, current, temperature=1., repeat_noise=False,
    )
    torch.testing.assert_close(predicted_clean, clean, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("discretization", ["uniform", "quad"])
@pytest.mark.parametrize("sampling_steps", [1, 3, 8])
def test_perfect_noise_sampling_reaches_clean_boundary(discretization, sampling_steps):
    model = diffusion_model(8)
    sampler = DDIMSampler(model, sampling_steps, ddim_discretize=discretization)
    clean = torch.tensor([[[[.5, -.8], [1.1, .3]]], [[[.2, .9], [-.7, 1.2]]]])
    terminal_alpha = model.alpha_bar[int(sampler.time_steps[-1])]
    epsilon = model.model.diffusion_model.noise.detach().expand_as(clean)
    initial = terminal_alpha.sqrt() * clean + (1. - terminal_alpha).sqrt() * epsilon
    actual = sampler.sample(list(clean.shape), cond=torch.zeros(2, 1, 1), x_last=initial)
    torch.testing.assert_close(actual, clean, rtol=2e-5, atol=2e-5)
    assert model.model.diffusion_model.seen_steps == [int(step) for step in reversed(sampler.time_steps) for _ in range(2)]


@pytest.mark.parametrize("discretization", ["uniform", "quad"])
@pytest.mark.parametrize("sampling_steps", [0, -1, 9])
def test_invalid_sampling_count_is_rejected_before_schedule_construction(discretization, sampling_steps):
    with pytest.raises(ValueError, match="n_steps must"):
        DDIMSampler(diffusion_model(8), sampling_steps, ddim_discretize=discretization)
