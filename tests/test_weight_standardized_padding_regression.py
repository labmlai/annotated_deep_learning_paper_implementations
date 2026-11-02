"""Weight-standardized Conv2d must preserve the parent layer's padding behavior."""
import pytest
import torch
import torch.nn.functional as F
from labml_nn.normalization.weight_standardization import weight_standardization
from labml_nn.normalization.weight_standardization.conv2d import Conv2d

def reference(layer, x):
    weight = weight_standardization(layer.weight, layer.eps)
    if layer.padding_mode != 'zeros':
        x = F.pad(x, layer._reversed_padding_repeated_twice, mode=layer.padding_mode)
        padding = 0
    else:
        padding = layer.padding
    return F.conv2d(x, weight, layer.bias, stride=layer.stride, padding=padding, dilation=layer.dilation, groups=layer.groups)

@pytest.mark.parametrize('mode', ['reflect', 'replicate', 'circular'])
@pytest.mark.parametrize('groups', [1, 2])
def test_nonzero_padding_matches_explicit_pad_then_convolution(mode, groups):
    torch.manual_seed(1)
    layer = Conv2d(2, 4, 3, padding=1, padding_mode=mode, groups=groups).double()
    x = torch.randn(2, 2, 5, 6, dtype=torch.float64)
    torch.testing.assert_close(layer(x), reference(layer, x), rtol=1e-12, atol=1e-12)

@pytest.mark.parametrize('mode', ['reflect', 'replicate', 'circular'])
def test_gradients_include_the_requested_boundary_mapping(mode):
    torch.manual_seed(2)
    layer = Conv2d(2, 2, 3, padding=1, padding_mode=mode).double()
    x = torch.randn(1, 2, 4, 5, dtype=torch.float64, requires_grad=True)
    targets = (x, layer.weight, layer.bias)
    observed = torch.autograd.grad(layer(x).square().sum(), targets)
    expected = torch.autograd.grad(reference(layer, x).square().sum(), targets)
    for actual, wanted in zip(observed, expected):
        torch.testing.assert_close(actual, wanted, rtol=1e-10, atol=1e-10)

def test_same_padding_with_even_kernel():
    torch.manual_seed(4)
    layer = Conv2d(2, 2, 2, padding='same', padding_mode='replicate').double()
    x = torch.randn(1, 2, 4, 5, dtype=torch.float64)
    torch.testing.assert_close(layer(x), reference(layer, x), rtol=1e-12, atol=1e-12)

@pytest.mark.parametrize('padding', [0, 1, 'same'])
def test_zero_padding_is_unchanged(padding):
    torch.manual_seed(3)
    layer = Conv2d(2, 2, 3, padding=padding).double()
    x = torch.randn(1, 2, 5, 5, dtype=torch.float64)
    torch.testing.assert_close(layer(x), reference(layer, x), rtol=1e-12, atol=1e-12)

def test_dilated_strided_nonzero_padding_without_bias():
    torch.manual_seed(9)
    layer = Conv2d(2, 4, 3, padding=2, dilation=2, stride=2, padding_mode='reflect', bias=False).double()
    x = torch.randn(1, 2, 7, 9, dtype=torch.float64)
    torch.testing.assert_close(layer(x), reference(layer, x), rtol=1e-12, atol=1e-12)
