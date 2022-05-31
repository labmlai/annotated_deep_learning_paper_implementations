"""
This is based on code by [@gharik](https://twitter.com/gharik).
"""

import random
import string
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from labml import monit, logger
from labml.configs import option
from labml.logger import Text
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs, transpose_batch


class ArithmeticDataset(Dataset):
    def __init__(self, seq_len: int, max_digits: int, n_sequences: int):
        self.n_sequences = n_sequences
        self.max_digits = max_digits
        self.seq_len = seq_len
        self.itos = list(string.digits + 'xe =\n?+;')
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @staticmethod
    def make_int(n_digits):
        res = 0
        for i in range(n_digits):
            d = random.randrange(1, 11) if i == 0 else random.randrange(0, 11)
            res = res * 10 + d

        return res

    @staticmethod
    def get_add_explanation(x, y):
        carry = 0
        e = 0
        explanation = []
        while x > 0 or y > 0 or carry > 0:
            rx, ry = x % 10, y % 10
            total = rx + ry + carry
            explanation.append(f"{rx}e{e}+{ry}e{e}+{carry}e{e}=={total}e{e}")
            x, y, c = x // 10, y // 10, total // 10
            e += 1

        return ' '.join(explanation)

    # Make a problem with a pre_explanation or not
    def make_add_problem(self):
        x = self.make_int(n_digits=random.randrange(1, self.max_digits + 1))
        y = self.make_int(n_digits=random.randrange(1, self.max_digits + 1))

        explanation = self.get_add_explanation(x, y)
        return f"x={x}+{y}; {explanation} x=={x + y}\n"

    def get_packed_math_input(self):
        s = ""
        s_enc = []
        while len(s_enc) <= self.seq_len:
            s_part = self.make_add_problem()
            s_part_enc = self.encode('?' + s_part)
            s_enc = s_enc + s_part_enc
        return s_enc

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, arr: List[int]):
        return ''.join([self.itos[c] for c in arr])

    def __getitem__(self, idx):
        s = torch.tensor(self.get_packed_math_input())
        return s[:self.seq_len], s[1:self.seq_len + 1]

    def __len__(self):
        return self.n_sequences


class ArithmeticAutoregression(NLPAutoRegressionConfigs):
    max_digits: int = 4
    train_sequences_per_epoch: int = 1024
    valid_sequences_per_epoch: int = 128
    train_loader: DataLoader = 'arithmetic_train_loader'
    valid_loader: DataLoader = 'arithmetic_valid_loader'

    n_tokens = len(ArithmeticDataset(1, 1, 1).itos)

    def sample(self):
        """
        ### Sampling function to generate samples periodically while training
        """

        # Starting prompt
        prompt = self.prompt
        # Collect output for printing
        log = [(prompt, Text.subtle)]
        # Dataset for decoding
        dataset = ArithmeticDataset(self.seq_len, self.max_digits, 1)
        # Sample 25 tokens
        for i in monit.iterate('Sample', self.seq_len - len(prompt)):
            # Tokenize the prompt
            data = torch.tensor(dataset.encode(prompt))[:, None]
            data = data.to(self.device)
            # Get the model output
            output, *_ = self.model(data)
            # Get the model prediction (greedy)
            output = output.argmax(dim=-1).squeeze()
            # Add the prediction to prompt
            prompt += self.prompt_separator + dataset.itos[output[-1]]
            # Add the prediction for logging
            log += [(self.prompt_separator + dataset.itos[output[-1]], Text.value)]

        # Print the sampled output
        logger.log(log)


@option(ArithmeticAutoregression.train_loader)
def arithmetic_train_loader(c: ArithmeticAutoregression):
    return DataLoader(ArithmeticDataset(c.seq_len, c.max_digits, c.train_sequences_per_epoch),
                      batch_size=c.batch_size,
                      collate_fn=transpose_batch)


@option(ArithmeticAutoregression.valid_loader)
def arithmetic_valid_loader(c: ArithmeticAutoregression):
    return DataLoader(ArithmeticDataset(c.seq_len, c.max_digits, c.valid_sequences_per_epoch),
                      batch_size=c.batch_size,
                      collate_fn=transpose_batch)


def _test():
    dataset = ArithmeticDataset(256, 8, 10)

    print(dataset.decode(dataset.get_packed_math_input()))


if __name__ == '__main__':
    _test()
