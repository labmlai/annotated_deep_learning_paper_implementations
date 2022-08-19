# GPT-NeoX

This is a simple implementation of [Eleuther GPT-NeoX](https://papers.labml.ai/paper/2204.06745) for inference and fine-tuning.


* [Model definition](https://nn.labml.ai/neox/model.html)
* [Tokenizer](https://nn.labml.ai/neox/tokenizer.html)
* [Checkpoint downloading and loading helpers](https://nn.labml.ai/neox/checkpoint.html)
* [Utilities](https://nn.labml.ai/neox/utils/index.html)

### [Samples](https://nn.labml.ai/neox/samples/__init__.py)

* [Generating text](https://nn.labml.ai/neox/samples/generate.html)
* [Fine tuning the biases with pipeline-parallel](https://nn.labml.ai/neox/samples/finetune.html)

### [Evaluation](https://nn.labml.ai/neox/evaluation/__init__.py)

* [Evaluating half precision model on a single GPU](https://nn.labml.ai/neox/evaluation/half_precision.html)

### Evaluation Results

| Task       | Metric          | NeoX Impl (2 GPU) | This repo (1 GPU) | LLM.Int8 |
|------------|-----------------|-------------------|-------------------|----------|
| anli_r1    | acc             | 0.3270            | 0.3360            | 0.3440   |
|            | acc_stderr      | 0.0148            | 0.0149            | 0.0150   |
| anli_r2    | acc             | 0.3410            | 0.3350            | 0.3540   |
|            | acc_stderr      | 0.0150            | 0.0149            | 0.0151   |
| anli_r3    | acc             | 0.3567            | 0.3525            | 0.3567   |
|            | acc_stderr      | 0.0138            | 0.0149            | 0.0138   |
| hellaswag  | acc             | 0.5351            | 0.5353            | 0.5348   |
|            | acc_stderr      | 0.0050            | 0.0050            | 0.0050   |
|            | acc_norm        | 0.7140            | 0.7145            | 0.7132   |
|            | acc_norm_stderr | 0.0045            | 0.0045            | 0.0045   |
| lambada    | acc             | 0.7211            | 0.7204            | 0.7155   |
|            | acc_stderr      | 0.0062            | 0.0063            | 0.0063   |
|            | ppl             | 3.6760            | 3.6375            | 3.7245   |
|            | ppl_stderr      | 0.0760            | 0.0747            | 0.0768   |
| piqa       | acc             | 0.7748            | 0.7758            | 0.7769   |
|            | acc_stderr      | 0.0097            | 0.0097            | 0.0097   |
|            | acc_norm        | 0.7786            | 0.7845            | 0.7829   |
|            | acc_norm_stderr | 0.0097            | 0.0096            | 0.0096   |
| winogrande | acc             | 0.6598            | 0.6582            | 0.6606   |
|            | acc_stderr      | 0.0133            | 0.0133            | 0.0133   |
| wsc        | acc             | 0.5096            | 0.5000            | 0.5288   |
|            | acc_stderr      | 0.0493            | 0.0493            | 0.0492   |
| mathqa     | acc             |                   |                   | 0.2720   |
|            | acc_stderr      |                   |                   | 0.0081   |
|            | acc_norm        |                   |                   | 0.2727   |
|            | acc_norm_stderr |                   |                   | 0.0082   |

**Official [Eleuther](https://www.eleuther.ai)
GPT-NoeX is source code is available at [eleutherai/gpt-neox](https://github.com/eleutherai/gpt-neox).**

