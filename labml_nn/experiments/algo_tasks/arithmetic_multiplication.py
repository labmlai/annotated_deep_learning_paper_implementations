"""
---
title: Arithmetic Dataset
summary: >
  This creates arithmetic problems.
---

*This is based on code by [Georges Harik (@gharik)](https://twitter.com/gharik).*
"""

import random
import string
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from labml import monit, logger, tracker
from labml.configs import option
from labml.logger import Text
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs, transpose_batch


class ArithmeticMultiplicationDataset(Dataset):
    """
    ## Arithmetic Dataset

    This creates arithmetic addition problems and solutions with workings.
    We've only implemented addition so far.

    It's based on a character level tokenization.
    """

    def __init__(self, seq_len: int, max_digits: int, base: int, n_sequences: int):
        """
        :param seq_len: is the sequence length of generated math problems.
            We fill as many problems as possible upto this length
        :max_digits: is the maximum number of digits in the operand integers
        :n_sequences: is the number of sequences per epoch
        """
        self.base = base
        self.n_sequences = n_sequences
        self.max_digits = max_digits
        self.seq_len = seq_len
        # Token id to string
        self.itos = list(string.digits + 'x =\n?*;')
        # Character to token id
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def make_int(self, n_digits: int):
        """
        Generates an integer with `n_digit` number of digits
        """
        res = 0
        for i in range(n_digits):
            d = random.randrange(1, self.base + 1) if i == 0 else random.randrange(0, self.base + 1)
            res = res * self.base + d

        return res

    def get_add_explanation(self, x: int, y: int):
        """
        Generates the workings for `x + y`.
        For example for `11+29` it generates
        `1e0+9e0+0e0=10e0 1e0+2e0+1e0=4e0`.
        """

        explanation = []
        while x > 0:
            rx = x % self.base
            explanation.append(f"{self.to_string(y * rx)}")
            x = x // self.base

        return ' '.join(explanation)

    # Make a problem with a pre_explanation or not
    def make_add_problem(self):
        """
        Creates an arithmetic addition problem with workings and answer.
        """
        x = self.make_int(n_digits=random.randrange(1, self.max_digits + 1))
        y = self.make_int(n_digits=random.randrange(1, self.max_digits + 1))

        explanation = self.get_add_explanation(x, y)
        return f"x={self.to_string(x)}*{self.to_string(y)}; {explanation} x=={self.to_string(x * y)}\n"

    def to_string(self, x: int):
        if x == 0:
            return '0'
        a = []
        while x > 0:
            a += [f'{x % self.base}']
            x = x // self.base

        return ''.join(reversed(a))

    def get_qa(self):
        """
        Get arithmetic problem and answer. This is used for evaluation.
        """
        x = self.make_int(n_digits=random.randrange(1, self.max_digits + 1))
        y = self.make_int(n_digits=random.randrange(1, self.max_digits + 1))

        return f'?x={self.to_string(x)}*{self.to_string(y)};', f'{self.to_string(x * y)}'

    def get_packed_math_input(self):
        """
        Generate multiple problems and pack them into a sequence.
        """
        s_enc = []
        mask = []
        while len(s_enc) <= self.seq_len:
            s_part = self.make_add_problem()
            s_part_enc = self.encode('?' + s_part)
            prob, sol = s_part.split(';')
            mask += [False] * (len(prob) + 2)
            mask += [True] * len(sol)
            s_enc = s_enc + s_part_enc
        return s_enc, mask

    def encode(self, s: str):
        """
        Encode a given string
        """
        return [self.stoi[c] for c in s]

    def decode(self, arr: List[int]):
        """
        Decode a list of token ids
        """
        return ''.join([self.itos[c] for c in arr])

    def __getitem__(self, idx: int):
        """
        Get a input and target pair for auto-regressive modelling
        """
        s, mask = self.get_packed_math_input()
        s = torch.tensor(s)
        mask = torch.tensor(mask)
        target = s * mask + -1 * (~mask)
        return s[:self.seq_len], target[1:self.seq_len + 1]

    def __len__(self):
        """
        Number of sequences per epoch
        """
        return self.n_sequences


class ArithmeticMultiplicationAutoregression(NLPAutoRegressionConfigs):
    """
    ## Arithmetic Task Experiment Configurations
    """
    # Maximum number of digits per operand integer
    max_digits: int = 4
    # Number of training sequences per epoch
    train_sequences_per_epoch: int = 2 ** 12
    # Training data loader
    train_loader: DataLoader = 'arithmetic_train_loader'
    # Number of problems in evaluation
    n_tests: int = 64
    # No need of a validation dataset
    validator = None
    # Number of times to run evaluations per epoch
    inner_iterations = 4
    # Number of tokens in the vocabulary
    base: int = 10
    n_tokens = len(ArithmeticMultiplicationDataset(1, 1, 1, 1).itos)

    @torch.no_grad()
    def sample(self):
        """
        ### Evaluation

        We use the sampling function to evaluate the model on a set of problems
        """

        # Skip in the first epoch
        if self.training_loop.idx < 1:
            return

        # Create a dataset to generate problems
        dataset = ArithmeticMultiplicationDataset(self.seq_len, self.max_digits, self.base, 1)
        # Get a set of problems and answers
        qa = [dataset.get_qa() for _ in range(self.n_tests)]
        # Collect the problems only
        questions = [p[0] for p in qa]

        # Create a tensor with only the initial token
        data = torch.tensor([[dataset.stoi[p[0]] for p in questions]])
        # Move to device
        data = data.to(self.device)

        # Number of sequences that have completed
        finished = torch.zeros((len(questions),)).bool().to(self.device)
        # Token id of the new line character - this marks end of the answer
        new_line = dataset.stoi['\n']

        # Sampled results
        results = [p[0] for p in questions]

        # Sample upto sequence length
        for i in monit.iterate('Sample', self.seq_len - 1):
            # If all the sequences have completed we skip this
            if finished.sum() == len(finished):
                continue

            # Get the model output
            output, *_ = self.model(data)
            # Get the model prediction (greedy)
            output = output[-1].argmax(dim=-1)

            # Override with the question
            for j, p in enumerate(questions):
                if len(p) > i + 1:
                    output[j] = dataset.stoi[p[i + 1]]

            # Find which sequences have finished
            finished = finished | (output == new_line)
            # Skip if all have finished
            if finished.sum() == len(finished):
                continue

            # Add the next token to the input
            data = torch.cat([data, output[None, :]], dim=0)

            # Get the sampled results
            for j, c in enumerate(output):
                results[j] += dataset.itos[c]

        # Discard everything after the answer in the results
        results = [r.split('\n')[0] for r in results]

        # Log a sample
        res_sample = results[0].split(';')
        logger.log([(res_sample[0], Text.key), (';', Text.subtle), (';'.join(res_sample[1:]), Text.none)])

        # Get the answers
        results = [r.split('x==')[-1] for r in results]

        # Count the number of correct answers
        correct = 0
        for r, _qa in zip(results, qa):
            if r == _qa[1]:
                correct += 1

        # Log the score
        tracker.save('score', correct / len(results))


@option(ArithmeticMultiplicationAutoregression.train_loader)
def arithmetic_train_loader(c: ArithmeticMultiplicationAutoregression):
    """
    Training data loader
    """
    return DataLoader(ArithmeticMultiplicationDataset(c.seq_len, c.max_digits, c.base, c.train_sequences_per_epoch),
                      batch_size=c.batch_size,
                      collate_fn=transpose_batch,
                      num_workers=4)


def _test():
    """
    Code to test generated problems
    """
    dataset = ArithmeticMultiplicationDataset(256, 4, 4, 10)

    print(dataset.decode(dataset.get_packed_math_input()[0]))


#
if __name__ == '__main__':
    _test()
