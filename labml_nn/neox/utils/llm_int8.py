"""
---
title: LLM.int8() on GPT-NeoX
summary: >
    Transform nn.Linear layers to 8-bit integer layers.
---

# LLM.int() on GPT-NeoX

This implements a utility function to transform a `nn.Linear` layer to LLM.int8() linear layer.

[LLM.int8() paper](https://papers.labml.ai/paper/eb2bcaee1d0011edaa66a71c10a887e7)
 shows you can use int8 quantization while handling outliers to
reduce memory footprint without performance degradation in large language models.
They convert weights and inputs to scaled 8-bit integers and does matrix multiplication
producing int32 results which is then converted back to float16 and rescaled.
They show that in large langauge models, some features can give extreme values (outliers)
that dominate the model's output.
These features get clamped in 8-bit integer space which causes the model performance to degrade.
As a solution they pick these outliers (greater than a specified threshold)
and compute their multiplications separately in float16 space.
Since the percentage of outliers is around 0.01% this doesn't increase memory usage,
and prevents the model from degrading performance.

The code to transform GPT-NoeX layers is defined in [model.py](../model.html#post_load_prepare).

Here are example uses of GPT-NeoX with int8 quantization.

* [Generate Text](../samples/llm_int8.html)
* [Run Evaluation Tests](../evaluation/llm_int8.html)
"""

# Import [`bitsandbytes`](https://github.com/timdettmers/bitsandbytes) package
try:
    from bitsandbytes.nn import Linear8bitLt, Int8Params
except ImportError:
    raise ImportError('''Please install `bitsandbytes` with `pip install bitsandbytes -U`''')

import torch
from torch import nn


def make_llm_int8_linear(linear_module: nn.Linear, device: torch.device, threshold: float = 6.0):
    """
    ## Transform a `nn.Linear` layer to LLM.int8() linear layer

    :param linear_module: is the `nn.Linear` layer to transform
    :param device: is the device of the model
    :param threshold: is the threshold $\alpha$ to use for outlier detection
    """

    #
    assert isinstance(linear_module, nn.Linear)

    # Create an empty Linear8bitLt module
    int8_lin = Linear8bitLt(
        linear_module.in_features,
        linear_module.out_features,
        linear_module.bias is not None,
        has_fp16_weights=False,
        threshold=threshold,
    )

    # Quantize the weights
    int8_lin._parameters['weight'] = Int8Params(linear_module.weight.data.cpu(),
                                                requires_grad=False,
                                                has_fp16_weights=False).to(device)

    # Set the bias in float16 space
    if linear_module.bias is not None:
        int8_lin._parameters['bias'] = nn.Parameter(linear_module.bias.data,
                                                    requires_grad=False)

    #
    return int8_lin
