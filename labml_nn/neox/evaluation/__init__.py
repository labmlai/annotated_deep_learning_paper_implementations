"""
---
title: Evaluation
summary: >
    Code to evaluate the model on NLP tasks through lm-evaluation-harness
---

# Evaluation

This is the code to test the model on
[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

* [Evaluating half precision model on a single GPU](half_precision.html)
"""
import math
from typing import List

import torch
import torch.nn.functional as F
from lm_eval import tasks, evaluator, utils
from lm_eval.base import BaseLM
from tokenizers import Tokenizer
from torch import nn
from tqdm import tqdm

from labml import monit
from labml_nn.neox.tokenizer import get_tokenizer


class EvalHarnessAdapter(BaseLM):
    """
    ## Evaluation Harness Adapter

    This is based on the [adapter from EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py)
    """

    def __init__(self, tokenizer: Tokenizer, vocab_size: int, batch_size: int):
        """
        :param tokenizer: is the [Huggingface Tokenizer](huggingface/tokenizers)
        :param vocab_size: is the size of the vocabulary
         (this differs from the tokenizer vocab size since neox adds some extra to make the embedding layer
         model parallel.)
        :param batch_size: is the batch size
        """
        super().__init__()
        self.tokenizer = tokenizer
        self._eot_token_id = self.tokenizer.token_to_id("<|endoftext|>")
        self._vocab_size = vocab_size

        self._batch_size = batch_size

    @property
    def device(self):
        raise RuntimeError()

    @property
    def vocab_size(self):
        """Size of the vocabulary"""
        return self._vocab_size

    @property
    def eot_token_id(self):
        """End-of-text token"""
        return self._eot_token_id

    @property
    def max_length(self):
        """Maximum sequence length"""
        return 2048

    @property
    def max_gen_toks(self):
        """Maximum number of tokens to generate"""
        return 128

    @property
    def batch_size(self):
        """
        Batch size
        """
        return self._batch_size

    def tok_encode(self, string: str):
        """
        Encode a given text
        """
        return self.tokenizer.encode(string).ids

    def tok_decode(self, tokens: List[int]):
        """
        Decode text from token ids
        """
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor):
        raise NotImplementedError

    def _model_generate(self, context, max_length, eos_token_id):
        raise RuntimeError()

    def greedy_until(self, requests):
        raise RuntimeError()

    @torch.no_grad()
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        """
        ### Get log-likelihoods of the next tokens

        :param requests: List of requests containing the context and the expected continuation.
        :param disable_tqdm: If True, disable tqdm progress bar.
        """

        # For results
        res = []

        # Reorder the requests in the descending order of the lengths,
        # so that sequences with similar lengths are close
        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        reord = utils.Reorderer(requests, _collate)

        # Loop through requests with `batch_size` number of requests at a time
        for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
            # To store the inputs for the batch
            inps = []
            # The continuations for the batch
            continuations = []
            # Lengths of the input sequences
            inplens = []
            # Padded length for the batch
            padded_length = None
            # Loop through each request in the chunk and collect them into PyTorch tensors with paddings
            for _, context_enc, continuation_enc in chunk:
                # Concatenate the context and continuation
                inp = context_enc + continuation_enc
                # Truncate from left if the size exceeds the `max_length`
                inp = inp[-(self.max_length + 1):]
                # Remove final token
                inp = inp[:-1]
                # Create a tensor
                inp = torch.tensor(inp, dtype=torch.long)
                # Input length
                inplen = inp.shape[0]

                # Determine the padded length.
                # Shorter sequences will get padded.
                if padded_length is None:
                    padded_length = int(math.ceil(inplen / 32)) * 32
                # padded_length = padded_length if padded_length is not None else inplen

                # Padding
                padding = torch.zeros(padded_length - inplen, dtype=torch.long)

                # Add padding
                inp = torch.cat([inp, padding], dim=0)

                inps.append(inp)
                continuations.append(continuation_enc)
                inplens.append(inplen)

            # Get model logits
            logits = self._model_call(torch.stack(inps))

            # Get log softmaxes
            multi_logits = F.log_softmax(logits, dim=-1)

            # Loop through the input/output pairs of the batch
            for logits, inplen, cont_toks in zip(multi_logits, inplens, continuations):
                # Get number of predicted tokens
                contlen = len(cont_toks)
                # Get logits of those
                logits = logits[inplen - contlen: inplen]
                # Get the tokens with the highest probabilities
                greedy_tokens = logits.argmax(dim=-1)
                # Get the target tokens
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).to(logits.device)
                # Whether there's an exact match
                max_equal = (greedy_tokens == cont_toks).all()
                # Log-likelihoods of the target tokens
                logits = torch.gather(logits, 1, cont_toks[:, None])
                # Add the total log-likelihoods and whether there was a match to the results
                res.append((float(logits.sum()), bool(max_equal)))

        # Re-order and return results
        return reord.get_original(res)

    @torch.no_grad()
    def run_eval(self, name: str, eval_tasks: List[str]):
        """
        ### Run given evaluations
        """

        # Run [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) evaluator
        results = evaluator.evaluate(lm=self, task_dict=tasks.get_task_dict(eval_tasks))

        # Add configs
        results["config"] = {
            "name": name,
        }

        #
        return results


class NoeXEvalHarnessAdapter(EvalHarnessAdapter):
    """
    ## Evaluation Harness Adapter

    This is based on the [adapter from EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py)
    """

    def __init__(self, model: nn.Module, tokenizer: Tokenizer, vocab_size: int, batch_size: int, device: torch.device):
        """
        :param model: is model
        :param tokenizer: is the [Huggingface Tokenizer](huggingface/tokenizers)
        :param vocab_size: is the size of the vocabulary
         (this differs from the tokenizer vocab size since neox adds some extra to make the embedding layer
         model parallel.)
        :param batch_size: is the batch size
        :param device: is the device of the model
        """
        super().__init__(tokenizer, vocab_size, batch_size)
        self.model = model
        self._device = device

    def _model_call(self, inps: torch.Tensor):
        """
        Call the model
        """
        return self.model(inps.to(self._device))


def run_eval_harness(model: nn.Module, name: str, eval_tasks: List[str], device: torch.device, batch_size: int = 8):
    """
    ## Run evaluation harness with a given model
    """

    # Load the tokenizer
    with monit.section('Load tokenizer'):
        tokenizer = get_tokenizer()

    # All tasks if nothing is specified
    if not eval_tasks:
        eval_tasks = [
            "anli_r1",
            "anli_r2",
            "anli_r3",
            "hellaswag",
            "lambada",
            "piqa",
            "winogrande",
            "wsc",
            "mathqa",
        ]

    # Create the adapter
    adapter = NoeXEvalHarnessAdapter(model, tokenizer, 50_432, batch_size, device)

    # Run
    return adapter.run_eval(name, eval_tasks)
