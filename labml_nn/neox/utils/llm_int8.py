"""
* [Generate](../samples/llm_int8.html)
* [Evaluation](../evaluation/llm_int8.html)
"""

try:
    from bitsandbytes.nn import Linear8bitLt, Int8Params
except ImportError:
    raise ImportError('''Please install `bitsandbytes` with `pip install bitsandbytes -U`''')

import torch
from torch import nn


def make_llm_int8_linear(linear_module: nn.Linear, device: torch.device, threshold: float = 6.0):
    # Create a Linear8bitLt module
    int8_lin = Linear8bitLt(
        linear_module.in_features,
        linear_module.out_features,
        linear_module.bias is not None,
        has_fp16_weights=False,
        threshold=threshold,
    )

    # Set the weights
    int8_lin._parameters['weight'] = Int8Params(linear_module.weight.data,
                                                requires_grad=False,
                                                has_fp16_weights=False).to(device)

    # Set the bias.
    # We don't have to convert this to Int8 since it doesn't use a lot of memory.
    # if linear_module.bias is not None:
    #     int8_lin._parameters['bias'] = nn.Parameter(linear_module.bias.data, requires_grad=False)

    return int8_lin
