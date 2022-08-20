"""
---
title: Generate Text with GPT-NeoX using LLM.int8() quantization
summary: >
     Generate Text with GPT-NeoX using LLM.int8() quantization
---

#  Generate Text with GPT-NeoX using LLM.int8() quantization

This shows how to generate text from GPT-NeoX using [LLM.int8() quantization](../utils/llm_int8.html).

This needs a GPU with 24GB memory.
"""

import torch
from torch import nn

from labml import monit
from labml_nn.neox.model import LayerGenerator
from labml_nn.neox.samples.generate import PROMPT, infer
from labml_nn.neox.utils import get_tokens, print_tokens
from labml_nn.neox.utils.cache import get_cache


def generate():
    """
    ## Generate text
    """

    # Setup [cache](../utils/cache.html) to cache intermediate key/value pairs for faster generation
    cache = get_cache()
    cache.set('use_cache', True)

    # Device
    device = torch.device('cuda:0')

    # Load layers in float16 into CPU. We convert the layers to int8 later, because doing that
    # on the fly after loading layers to GPU causes CUDA memory fragmentation
    # (about 3GB memory can get lost due to fragmentation).
    layer_generator = LayerGenerator(is_clone_layers=True,
                                     dtype=torch.float16,
                                     device=torch.device('cpu'),
                                     is_llm_int8=False,
                                     )
    layers = list(layer_generator.load())

    # This reduces CUDA memory fragmentation
    for layer in monit.iterate('Convert to int8', layers, is_children_silent=True):
        layer_generator.post_load_prepare(layer,
                                          device=device,
                                          is_llm_int8=True,
                                          llm_int8_threshold=6.0,
                                          )
        layer.to(device)

    # Create `nn.Sequential` model
    model = nn.Sequential(*layers)

    # Clear cache and print memory summary for debugging
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary())

    # Get token ids
    ids = get_tokens(PROMPT)

    # Run the model.
    # We use the [`infer`](generate.html) function defined in [`generate.py`](generate.html)
    cache.set('state_ids', (None, 1))
    with monit.section('Infer'):
        next_token = infer(model, ids, device)[-1]

    # Append the predicted token
    ids += [next_token]

    # Predict 100 tokens
    for i in range(1, 100):
        # Set the state to use cached activations
        cache.set('state_ids', (i, i + 1))
        # Get next token. Note that we only feed the last token to the model because
        # we cache the key/value pairs of previous tokens.
        with monit.section('Infer'):
            next_token = infer(model, [next_token], device)[-1]
        # Append the predicted token
        ids += [next_token]
        # Print
        print_tokens(ids, [ids])


#
if __name__ == '__main__':
    generate()
