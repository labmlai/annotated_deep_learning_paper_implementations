"""DDPM attention uses normalized features and preserves the original residual."""

import copy

import pytest
import torch
import torch.nn.functional as F

from labml_nn.diffusion.ddpm import DenoiseDiffusion
from labml_nn.diffusion.ddpm.unet import AttentionBlock, UNet


def reference_attention(block, x):
    """Independent SDPA contraction after functional image-space GroupNorm."""
    batch, channels, height, width = x.shape
    normalized = F.group_norm(x, block.norm.num_groups, block.norm.weight,
                              block.norm.bias, block.norm.eps)
    tokens = normalized.flatten(2).transpose(1, 2)
    projected = F.linear(tokens, block.projection.weight, block.projection.bias)
    projected = projected.reshape(batch, height * width, block.n_heads, 3 * block.d_k)
    q, k, v = (part.transpose(1, 2) for part in projected.chunk(3, dim=-1))
    attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
    attended = attended.transpose(1, 2).reshape(batch, height * width, -1)
    branch = F.linear(attended, block.output.weight, block.output.bias)
    return x + branch.transpose(1, 2).reshape(batch, channels, height, width)


@pytest.mark.parametrize("heads,head_width,groups,dtype", [
    (1, None, 32, torch.float32),
    (1, 8, 8, torch.float64),
    (1, 5, 1, torch.bfloat16),
])
def test_normalized_attention_matches_sdpa_outputs_and_gradients(heads, head_width, groups, dtype):
    torch.manual_seed(510)
    block = AttentionBlock(32, heads, head_width, groups).to(dtype=dtype)
    with torch.no_grad():
        block.norm.weight.copy_(torch.linspace(.6, 1.4, 32))
        block.norm.bias.copy_(torch.linspace(-.3, .4, 32))
    reference = copy.deepcopy(block)
    x = (torch.randn(2, 32, 2, 3, dtype=dtype) * 1.7 + 2.).requires_grad_()
    expected_x = x.detach().clone().requires_grad_()
    actual = block(x)
    expected = reference_attention(reference, expected_x)
    tolerance = dict(rtol=.04, atol=.04) if dtype == torch.bfloat16 else dict(rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual, expected, **tolerance)
    weights = torch.linspace(-.8, 1.2, actual.numel(), dtype=dtype).reshape_as(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(x.grad, expected_x.grad, **tolerance)
    for (name, parameter), (reference_name, reference_parameter) in zip(
            block.named_parameters(), reference.named_parameters()):
        assert name == reference_name
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, **tolerance)


@pytest.mark.parametrize("heads,groups", [(1, 32), (1, 4)])
def test_group_offsets_pass_only_through_the_residual(heads, groups):
    torch.manual_seed(511)
    block = AttentionBlock(32, n_heads=heads, d_k=8, n_groups=groups).double()
    x = torch.randn(2, 32, 3, 2, dtype=torch.float64)
    offsets = torch.randn(2, groups, 1, 1, dtype=torch.float64).repeat_interleave(32 // groups, dim=1) * 3
    shifted = block(x + offsets)
    ordinary = block(x)
    # GroupNorm cancels each group's constant shift in the attention branch.
    torch.testing.assert_close(shifted - ordinary, offsets.expand_as(x), rtol=1e-10, atol=1e-10)


def test_attention_normalization_affine_parameters_are_trainable():
    torch.manual_seed(512)
    block = AttentionBlock(32, n_heads=1, d_k=8, n_groups=8)
    x = torch.randn(2, 32, 3, 3)
    block(x).square().mean().backward()
    for parameter in (block.norm.weight, block.norm.bias):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("groups", [1, 32])
def test_zero_attention_projection_preserves_unnormalized_residual(groups):
    torch.manual_seed(513)
    block = AttentionBlock(32, n_heads=1, d_k=8, n_groups=groups).double()
    with torch.no_grad():
        block.norm.weight.fill_(2.)
        block.norm.bias.fill_(3.)
        block.output.weight.zero_()
        block.output.bias.zero_()
    x = (torch.randn(2, 32, 2, 3, dtype=torch.float64) * 4 + 8).requires_grad_()
    actual = block(x)
    torch.testing.assert_close(actual, x, rtol=0., atol=0.)
    actual.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0., atol=0.)


def test_native_unet_trains_every_attention_normalization():
    torch.manual_seed(514)
    model = UNet(image_channels=3, n_channels=32, ch_mults=(1, 2),
                 is_attn=(True, True), n_blocks=1)
    image = torch.randn(2, 3, 4, 6)
    diffusion = DenoiseDiffusion(model, n_steps=8, device=torch.device("cpu"))
    loss = diffusion.loss(image, noise=torch.randn_like(image))
    assert torch.isfinite(loss)
    loss.backward()
    attention_blocks = [block for block in model.modules() if isinstance(block, AttentionBlock)]
    assert len(attention_blocks) == 7
    for block in attention_blocks:
        for parameter in (block.norm.weight, block.norm.bias):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
