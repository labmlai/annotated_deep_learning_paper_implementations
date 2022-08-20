"""
---
title: GPT-NeoX
summary: >
    Simple GPT-NeoX implementation
---

# GPT-NeoX

This is a simple implementation of [Eleuther GPT-NeoX](https://papers.labml.ai/paper/2204.06745) for inference and fine-tuning.


* [Model definition](model.html)
* [Tokenizer](tokenizer.html)
* [Checkpoint downloading and loading helpers](checkpoint.html)
* [Utilities](utils/index.html)
* [LLM.int8() quantization](utils/llm_int8.html)

### [Samples](samples/__init__.py)

* [Generating text](samples/generate.html)
* [Fine-tuning the biases with pipeline-parallel](samples/finetune.html)
* [Generating text with LLM.int8()](samples/llm_int8.html)

### [Evaluation](evaluation/__init__.py)

* [Evaluating half precision model on a single GPU](evaluation/half_precision.html)
* [Evaluating LLM.int8() model](evaluation/llm_int8.html)

**Official [Eleuther](https://www.eleuther.ai)
GPT-NoeX is source code is available at [eleutherai/gpt-neox](https://github.com/eleutherai/gpt-neox).**
"""
