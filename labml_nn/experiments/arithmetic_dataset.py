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
from labml.logger import Text
from torch.utils.data import DataLoader, Dataset

from labml import monit, logger, tracker
from labml.configs import option
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs, transpose_batch


class ArithmeticDataset(Dataset):
    """
    ## Arithmetic Dataset

    This creates arithmetic addition problems and solutions with workings.
    We've only implemented addition so far.

    It's based on a character level tokenization.
    """

    def __init__(self, seq_len: int, max_digits: int, n_sequences: int):
        """
        :param seq_len: is the sequence length of generated math problems.
            We fill as many problems as possible upto this length
        :max_digits: is the maximum number of digits in the operand integers
        :n_sequences: is the number of sequences per epoch
        """
        self.n_sequences = n_sequences
        self.max_digits = max_digits
        self.seq_len = seq_len
        # Token id to string
        self.itos = list(string.digits + 'xe =\n?+;')
        # Character to token id
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @staticmethod
    def make_int(n_digits: int):
        """
        Generates an integer with `n_digit` number of digits
        """
        res = 0
        for i in range(n_digits):
            d = random.randrange(1, 11) if i == 0 else random.randrange(0, 11)
            res = res * 10 + d

        return res

    @staticmethod
    def get_add_explanation(x: int, y: int):
        """
        Generates the workings for `x + y`.
        For example for `11+29` it generates
        `1e0+9e0+0e0=10e0 1e0+2e0+1e0=4e0`.
        """

        carry = 0
        e = 0
        explanation = []
        while x > 0 or y > 0 or carry > 0:
            rx, ry = x % 10, y % 10
            total = rx + ry + carry
            explanation.append(f"{rx}e{e}+{ry}e{e}+{carry}e{e}=={total}e{e}")
            x, y, carry = x // 10, y // 10, total // 10
            e += 1

        return ' '.join(explanation)

    # Make a problem with a pre_explanation or not
    def make_add_problem(self):
        """
        Creates an arithmetic addition problem with workings and answer.
        """
        x = self.make_int(n_digits=random.randrange(1, self.max_digits + 1))
        y = self.make_int(n_digits=random.randrange(1, self.max_digits + 1))

        explanation = self.get_add_explanation(x, y)
        return f"x={x}+{y}; {explanation} x=={x + y}\n"

    def get_qa(self):
        """
        Get arithmetic problem and answer. This is used for evaluation.
        """
        x = self.make_int(n_digits=random.randrange(1, self.max_digits + 1))
        y = self.make_int(n_digits=random.randrange(1, self.max_digits + 1))

        return f'x={x}+{y};', f'{x + y}'

    def get_packed_math_input(self):
        """
        Generate multiple problems and pack them into a sequence.
        """
        s_enc = []
        while len(s_enc) <= self.seq_len:
            s_part = self.make_add_problem()
            s_part_enc = self.encode('?' + s_part)
            s_enc = s_enc + s_part_enc
        return s_enc

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
        s = torch.tensor(self.get_packed_math_input())
        return s[:self.seq_len], s[1:self.seq_len + 1]

    def __len__(self):
        """
        Number of sequences per epoch
        """
        return self.n_sequences


class ArithmeticAutoregression(NLPAutoRegressionConfigs):
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
    n_tokens = len(ArithmeticDataset(1, 1, 1).itos)

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
        dataset = ArithmeticDataset(self.seq_len, self.max_digits, 1)
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

            # Find which sequences have finished
            finished = finished | (output == new_line)
            # Skip if all have finished
            if finished.sum() == len(finished):
                continue

            # Override with the question
            for j, p in enumerate(questions):
                if len(p) > i + 1:
                    output[j] = dataset.stoi[p[i + 1]]

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


@option(ArithmeticAutoregression.train_loader)
def arithmetic_train_loader(c: ArithmeticAutoregression):
    """
    Training data loader
    """
    return DataLoader(ArithmeticDataset(c.seq_len, c.max_digits, c.train_sequences_per_epoch),
                      batch_size=c.batch_size,
                      collate_fn=transpose_batch,
                      num_workers=4)


def _test():
    """
    Code to test generated problems
    """
    dataset = ArithmeticDataset(256, 8, 10)

    print(dataset.decode(dataset.get_packed_math_input()))


#
if __name__ == '__main__':
    _test()
