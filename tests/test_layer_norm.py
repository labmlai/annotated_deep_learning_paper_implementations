"""Numerical regressions for the annotated layer normalization implementation."""

import pytest
import torch
from torch.nn import functional as F

from labml_nn.normalization.layer_norm import LayerNorm


@pytest.mark.parametrize('dtype', [pytest.param(torch.float32, id='float32'), pytest.param(torch.float64, id='float64')])
@pytest.mark.parametrize('offset', [0.0, 10000.0, 100000.0])
def test_layer_norm_is_stable_under_common_offset(dtype, offset):
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=dtype) + offset
    layer = LayerNorm(4, elementwise_affine=False)
    expected = F.layer_norm(x.double(), (4,), eps=layer.eps).to(dtype)

    actual = layer(x)

    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('dtype', [pytest.param(torch.float16, id='float16'), pytest.param(torch.bfloat16, id='bfloat16')])
@pytest.mark.parametrize('affine', [False, True])
@pytest.mark.parametrize('values', [[-60000.0, -20000.0, 20000.0, 60000.0],
                                  [1000.0, 1008.0, 1016.0, 1024.0]])
def test_layer_norm_accumulates_low_precision_statistics_in_float32(dtype, affine, values):
    x = torch.tensor([values], dtype=dtype)
    layer = LayerNorm(4, elementwise_affine=affine).to(dtype)
    expected = F.layer_norm(x.double(), (4,), eps=layer.eps).to(dtype)

    actual = layer(x)

    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('shape', [torch.Size([1]), torch.Size([2, 3])])
@pytest.mark.parametrize('affine', [False, True])
def test_layer_norm_population_variance_and_affine_gradients(shape, affine):
    x = torch.arange(2 * shape.numel(), dtype=torch.float64).reshape(2, *shape)
    x = (x + 10000).requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    layer = LayerNorm(shape, elementwise_affine=affine).double()
    gain = bias = None
    if affine:
        with torch.no_grad():
            layer.gain.copy_(torch.linspace(0.5, 1.5, shape.numel()).reshape(shape))
            layer.bias.copy_(torch.linspace(-0.2, 0.2, shape.numel()).reshape(shape))
        gain = layer.gain.detach().clone().requires_grad_()
        bias = layer.bias.detach().clone().requires_grad_()
    weights = torch.linspace(-1, 2, x.numel(), dtype=torch.float64).reshape(x.shape)

    actual = layer(x)
    expected = F.layer_norm(ref_x, shape, gain, bias, layer.eps)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=1e-8, atol=1e-9)
    if shape.numel() == 1:
        # A singleton feature has exactly zero centered value and variance.
        # Some native kernels leave a tiny cancellation residual for d(gain).
        assert torch.count_nonzero(x.grad) == 0
        if affine:
            torch.testing.assert_close(actual, layer.bias.expand_as(actual), rtol=0, atol=0)
            assert torch.count_nonzero(layer.gain.grad) == 0
        else:
            assert torch.count_nonzero(actual) == 0
    if affine:
        gain_atol = 1e-9 if shape.numel() == 1 else 1e-10
        torch.testing.assert_close(layer.gain.grad, gain.grad, rtol=1e-10, atol=gain_atol)
        torch.testing.assert_close(layer.bias.grad, bias.grad, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('dtype', [pytest.param(torch.float16, id='float16'), pytest.param(torch.bfloat16, id='bfloat16')])
def test_layer_norm_preserves_affine_type_promotion(dtype):
    x = torch.tensor([[-1000.0, -500.0, 500.0, 1000.0]], dtype=dtype)
    layer = LayerNorm(4)
    normalized = F.layer_norm(x.double(), (4,), eps=layer.eps).to(dtype)
    expected = layer.gain * normalized + layer.bias

    actual = layer(x)

    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_layer_norm_gradcheck():
    x = torch.tensor([[10000.0, 10001.0, 10002.0, 10003.0]],
                     dtype=torch.float64, requires_grad=True)
    layer = LayerNorm(4, elementwise_affine=False)

    assert torch.autograd.gradcheck(layer, (x,))
