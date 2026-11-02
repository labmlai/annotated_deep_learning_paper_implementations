"""Cache and mixed-precision regressions for rotary positional embeddings."""

import pytest
import torch

from labml_nn.transformers.rope import RotaryPositionalEmbeddings, RotaryPEMultiHeadAttention
from labml_nn.transformers.rope.value_pe import ReverseRotaryPositionalEmbeddings, RotaryValuePEMultiHeadAttention


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize('rotated_features', [4, 8])
@pytest.mark.parametrize('rotation', [RotaryPositionalEmbeddings, ReverseRotaryPositionalEmbeddings])
def test_rope_preserves_input_dtype_and_passthrough(dtype, rotated_features, rotation):
    x = torch.arange(48, dtype=dtype).reshape(3, 1, 2, 8) / 16
    rope = rotation(rotated_features)

    actual = rope(x)

    assert actual.dtype == dtype
    assert rope.cos_cached.dtype == torch.float32
    assert rope.sin_cached.dtype == torch.float32
    torch.testing.assert_close(actual[0], x[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[..., rotated_features:], x[..., rotated_features:], rtol=0, atol=0)
    expected_norms = torch.linalg.vector_norm(x[..., :rotated_features].double(), dim=-1)
    actual_norms = torch.linalg.vector_norm(actual[..., :rotated_features].double(), dim=-1)
    tolerance = 0.01 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(actual_norms, expected_norms, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize('rotation', [RotaryPositionalEmbeddings, ReverseRotaryPositionalEmbeddings])
def test_rope_reuses_and_extends_cache_across_dtype_changes(dtype, rotation):
    rope = rotation(4)
    warmup = torch.ones(6, 1, 1, 4)
    rope(warmup)
    cached = rope.cos_cached
    x = torch.ones(3, 1, 1, 4, dtype=dtype)

    actual = rope(x)
    expected = rotation(4)(x)

    assert rope.cos_cached is cached
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    long_x = torch.ones(9, 1, 1, 4, dtype=dtype)
    torch.testing.assert_close(rope(long_x), rotation(4)(long_x), rtol=0, atol=0)


@pytest.mark.parametrize('rotation', [RotaryPositionalEmbeddings, ReverseRotaryPositionalEmbeddings])
def test_rope_rebuilds_cache_for_input_device(rotation):
    rope = rotation(4)
    # A meta tensor permits testing stale-device cache handling without a GPU.
    rope(torch.empty(6, 1, 1, 4, device='meta'))
    assert rope.cos_cached.device.type == 'meta'
    x = torch.ones(3, 1, 1, 4)

    actual = rope(x)

    assert actual.device == x.device
    assert rope.cos_cached.device == x.device
    torch.testing.assert_close(actual, rotation(4)(x), rtol=0, atol=0)


@pytest.mark.parametrize('attention_type', [RotaryPEMultiHeadAttention, RotaryValuePEMultiHeadAttention])
def test_rope_half_precision_attention_forward_and_backward(attention_type):
    attention = attention_type(2, 8, dropout_prob=0).bfloat16()
    x = torch.arange(32, dtype=torch.bfloat16).reshape(4, 1, 8).requires_grad_()

    actual = attention(query=x, key=x, value=x)
    actual.sum().backward()

    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    assert torch.isfinite(x.grad).all()
    assert attention.attn.dtype == torch.bfloat16


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_rope_attention_depends_on_relative_position(dtype):
    query = torch.tensor([1.0, -0.5, 2.0, 0.75], dtype=dtype).repeat(8, 1, 1, 1)
    key = torch.tensor([-0.25, 1.5, 0.5, 2.0], dtype=dtype).repeat(8, 1, 1, 1)
    rope = RotaryPositionalEmbeddings(4)
    scores = torch.einsum('ibhd,jbhd->ijbh', rope(query), rope(key))

    torch.testing.assert_close(scores[:-1, :-1], scores[1:, 1:], rtol=1e-5, atol=1e-6)


def test_rope_gradcheck():
    x = torch.arange(12, dtype=torch.float64).reshape(3, 1, 1, 4).requires_grad_()

    assert torch.autograd.gradcheck(RotaryPositionalEmbeddings(4), (x,))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
def test_rope_inverse_rotation_recovers_input(dtype):
    x = torch.arange(24, dtype=dtype).reshape(6, 1, 1, 4) / 16
    forward = RotaryPositionalEmbeddings(4)
    reverse = ReverseRotaryPositionalEmbeddings(4)

    actual = reverse(forward(x))

    assert actual.dtype == dtype
    tolerance = 0.02 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(actual, x, rtol=tolerance, atol=tolerance)
