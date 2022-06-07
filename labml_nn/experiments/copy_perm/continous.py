import random
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from labml import tracker
from labml.configs import option
from labml_helpers.train_valid import BatchIndex
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs, transpose_batch


class CopyPermRepeatDataset(Dataset):
    """
    """

    def __init__(self, seq_len: int, substr_len: int, rnd_len: int, n_sequences: int):
        """
        :param seq_len: is the sequence length of generated math problems.
            We fill as many problems as possible upto this length
        """
        self.rnd_len = rnd_len
        self.substr_len = substr_len
        self.n_sequences = n_sequences
        self.seq_len = seq_len
        self.letters = 'acgt'  # string.ascii_lowercase  # '01'  # 'acgt'  #
        # Token id to string
        self.itos = list(self.letters + '>')
        # Character to token id
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def random_string(self, n_chars: int):
        return ''.join(random.choice(self.letters) for _ in range(n_chars))

    def generate_problem(self):
        pure = self.random_string(self.substr_len)
        out = pure
        mask = [False] * len(out)
        while len(out) <= self.seq_len:
            s = self.random_string(random.randrange(1, self.rnd_len))
            out += s + '>'
            mask += [False] * (len(s) + 1)
            pure += s

            offset = random.randrange(0, len(pure) - self.substr_len)
            copy = pure[offset:offset + self.substr_len]

            out += copy
            mask += [False] + [True] * (self.substr_len - 1)
            pure += copy

        return out, mask

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
        s, mask = self.generate_problem()
        s = torch.tensor(self.encode(s))
        mask = torch.tensor(mask)
        target = s * mask + -1 * (~mask)
        return s[:self.seq_len], target[1:self.seq_len + 1]

    def __len__(self):
        """
        Number of sequences per epoch
        """
        return self.n_sequences


class CopyRepeatAutoregression(NLPAutoRegressionConfigs):
    """
    ## Arithmetic Task Experiment Configurations
    """
    # Number of training sequences per epoch
    train_sequences_per_epoch: int = 2 ** 12
    # Training data loader
    train_loader: DataLoader = 'copy_train_loader'
    # Number of problems in evaluation
    n_tests: int = 64
    # No need of a validation dataset
    validator = None
    # Number of times to run evaluations per epoch
    inner_iterations = 4
    # Number of tokens in the vocabulary
    n_tokens = len(CopyPermRepeatDataset(1, 1, 1, 1).itos)

    substr_len: int = 16
    rnd_len: int = 16

    @torch.no_grad()
    def sample(self):
        pass

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        ### Training or validation step
        """

        # Set training/eval mode
        self.model.train(self.mode.is_train)

        # Move data to the device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Update global step (number of tokens processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        # Whether to capture model outputs
        with self.mode.update(is_log_activations=batch_idx.is_last and self.is_log_model_activations):
            # Get model outputs.
            # It's returning a tuple for states when using RNNs.
            # This is not implemented yet. 😜
            output, *_ = self.model(data)

        # Calculate and log loss
        loss = self.loss_func(output, target)
        tracker.add("loss.", loss)

        # Calculate and log accuracy
        self.accuracy(output, target)
        self.accuracy.track()

        self.other_metrics(output, target)

        # Train the model
        if self.mode.is_train:
            # Calculate gradients
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            # Take optimizer step
            self.optimizer.step()
            # Log the model parameters and gradients on last batch of every epoch
            if batch_idx.is_last and self.is_log_model_params_grads:
                tracker.add('model', self.model)
            # Clear the gradients
            self.optimizer.zero_grad()

        # Save the tracked metrics
        tracker.save()


@option(CopyRepeatAutoregression.train_loader)
def copy_train_loader(c: CopyRepeatAutoregression):
    """
    Training data loader
    """
    return DataLoader(CopyPermRepeatDataset(c.seq_len, c.substr_len, c.rnd_len, c.train_sequences_per_epoch),
                      batch_size=c.batch_size,
                      collate_fn=transpose_batch)
    # num_workers=4)


def _test():
    """
    Code to test generated problems
    """
    dataset = CopyPermRepeatDataset(32, 8, 8, 1)

    print(dataset.generate_problem())


#
if __name__ == '__main__':
    _test()
