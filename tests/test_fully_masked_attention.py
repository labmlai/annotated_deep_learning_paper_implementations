"""Masked queries must remain finite and obey the zero-attention contract."""

import pytest
import torch
import torch.nn.functional as F

from labml_nn.transformers.mha import MultiHeadAttention


def sdpa_oracle(module, query, key, value, mask):
    projected = []
    for layer, tensor in [(module.query.linear, query), (module.key.linear, key),
                          (module.value.linear, value)]:
        projection = F.linear(tensor, layer.weight, layer.bias)
        projected.append(projection.reshape(tensor.shape[0], tensor.shape[1], module.heads, module.d_k)
                         .permute(1, 2, 0, 3))
    sdpa_mask = None if mask is None else mask.bool().permute(2, 0, 1).unsqueeze(1)
    result = F.scaled_dot_product_attention(*projected, attn_mask=sdpa_mask, dropout_p=0.)
    result = result.permute(2, 0, 1, 3).reshape(query.shape[0], query.shape[1], -1)
    return F.linear(result, module.output.weight, module.output.bias)


def attention_mask(pattern):
    mask = torch.ones(4, 5, 2, dtype=torch.bool)
    if pattern == "query_holes":
        mask[1] = False
        mask[3, :, 1] = False
        mask[0, 3:] = False
    elif pattern == "empty_batch":
        mask[:, :, 0] = False
        mask[2, 1:, 1] = False
    elif pattern == "broadcast_query":
        mask = mask[:1]
        mask[:, :, 0] = False
        mask[:, 3:, 1] = False
    elif pattern == "broadcast_batch":
        mask = mask[:, :, :1]
        mask[2] = False
        mask[0, 2:] = False
    elif pattern == "no_mask":
        return None
    return mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("pattern", ["query_holes", "empty_batch", "broadcast_query", "broadcast_batch", "all_valid", "no_mask"])
def test_masked_attention_matches_sdpa_values_and_gradients(dtype, bias, pattern):
    torch.manual_seed(31)
    module = MultiHeadAttention(heads=2, d_model=8, dropout_prob=0., bias=bias).to(dtype)
    originals = [torch.randn(length, 2, 8, dtype=dtype) * .3 for length in (4, 5, 5)]
    actual_inputs = [tensor.clone().requires_grad_() for tensor in originals]
    reference_inputs = [tensor.clone().requires_grad_() for tensor in originals]
    mask = attention_mask(pattern)
    actual = module(query=actual_inputs[0], key=actual_inputs[1], value=actual_inputs[2], mask=mask)
    expected = sdpa_oracle(module, *reference_inputs, mask)
    tolerance = .02 if dtype == torch.bfloat16 else (2e-5 if dtype == torch.float32 else 2e-10)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    weights = torch.linspace(-.8, .9, actual.numel(), dtype=dtype).reshape_as(actual)
    parameters = tuple(module.parameters())
    expected_gradients = torch.autograd.grad((expected * weights).sum(), (*reference_inputs, *parameters))
    actual_gradients = torch.autograd.grad((actual * weights).sum(), (*actual_inputs, *parameters))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=tolerance, atol=tolerance)
    if mask is not None:
        empty_rows = ~mask.any(dim=1, keepdim=True).unsqueeze(-1)
        masked_probabilities = module.attn.masked_select(empty_rows.expand_as(module.attn))
        assert torch.count_nonzero(masked_probabilities) == 0
