import random
from pathlib import PurePath, Path
from typing import List, Callable, Dict, Optional

from torchvision import datasets, transforms

import torch
from labml import lab
from labml import monit
from labml.configs import BaseConfigs
from labml.configs import aggregate, option
from labml.utils.download import download_file
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset, Dataset


def _mnist_dataset(is_train, transform):
    return datasets.MNIST(str(lab.get_data_path()),
                          train=is_train,
                          download=True,
                          transform=transform)


class MNISTConfigs(BaseConfigs):
    """
    Configurable MNIST data set.

    Arguments:
        dataset_name (str): name of the data set, ``MNIST``
        dataset_transforms (torchvision.transforms.Compose): image transformations
        train_dataset (torchvision.datasets.MNIST): training dataset
        valid_dataset (torchvision.datasets.MNIST): validation dataset

        train_loader (torch.utils.data.DataLoader): training data loader
        valid_loader (torch.utils.data.DataLoader): validation data loader

        train_batch_size (int): training batch size
        valid_batch_size (int): validation batch size

        train_loader_shuffle (bool): whether to shuffle training data
        valid_loader_shuffle (bool): whether to shuffle validation data
    """

    dataset_name: str = 'MNIST'
    dataset_transforms: transforms.Compose
    train_dataset: datasets.MNIST
    valid_dataset: datasets.MNIST

    train_loader: DataLoader
    valid_loader: DataLoader

    train_batch_size: int = 64
    valid_batch_size: int = 1024

    train_loader_shuffle: bool = True
    valid_loader_shuffle: bool = False


@option(MNISTConfigs.dataset_transforms)
def mnist_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


@option(MNISTConfigs.train_dataset)
def mnist_train_dataset(c: MNISTConfigs):
    return _mnist_dataset(True, c.dataset_transforms)


@option(MNISTConfigs.valid_dataset)
def mnist_valid_dataset(c: MNISTConfigs):
    return _mnist_dataset(False, c.dataset_transforms)


@option(MNISTConfigs.train_loader)
def mnist_train_loader(c: MNISTConfigs):
    return DataLoader(c.train_dataset,
                      batch_size=c.train_batch_size,
                      shuffle=c.train_loader_shuffle)


@option(MNISTConfigs.valid_loader)
def mnist_valid_loader(c: MNISTConfigs):
    return DataLoader(c.valid_dataset,
                      batch_size=c.valid_batch_size,
                      shuffle=c.valid_loader_shuffle)


aggregate(MNISTConfigs.dataset_name, 'MNIST',
          (MNISTConfigs.dataset_transforms, 'mnist_transforms'),
          (MNISTConfigs.train_dataset, 'mnist_train_dataset'),
          (MNISTConfigs.valid_dataset, 'mnist_valid_dataset'),
          (MNISTConfigs.train_loader, 'mnist_train_loader'),
          (MNISTConfigs.valid_loader, 'mnist_valid_loader'))


def _cifar_dataset(is_train, transform):
    return datasets.CIFAR10(str(lab.get_data_path()),
                            train=is_train,
                            download=True,
                            transform=transform)


class CIFAR10Configs(BaseConfigs):
    """
    Configurable CIFAR 10 data set.

    Arguments:
        dataset_name (str): name of the data set, ``CIFAR10``
        dataset_transforms (torchvision.transforms.Compose): image transformations
        train_dataset (torchvision.datasets.CIFAR10): training dataset
        valid_dataset (torchvision.datasets.CIFAR10): validation dataset

        train_loader (torch.utils.data.DataLoader): training data loader
        valid_loader (torch.utils.data.DataLoader): validation data loader

        train_batch_size (int): training batch size
        valid_batch_size (int): validation batch size

        train_loader_shuffle (bool): whether to shuffle training data
        valid_loader_shuffle (bool): whether to shuffle validation data
    """
    dataset_name: str = 'CIFAR10'
    dataset_transforms: transforms.Compose
    train_dataset: datasets.CIFAR10
    valid_dataset: datasets.CIFAR10

    train_loader: DataLoader
    valid_loader: DataLoader

    train_batch_size: int = 64
    valid_batch_size: int = 1024

    train_loader_shuffle: bool = True
    valid_loader_shuffle: bool = False


@CIFAR10Configs.calc(CIFAR10Configs.dataset_transforms)
def cifar10_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


@CIFAR10Configs.calc(CIFAR10Configs.train_dataset)
def cifar10_train_dataset(c: CIFAR10Configs):
    return _cifar_dataset(True, c.dataset_transforms)


@CIFAR10Configs.calc(CIFAR10Configs.valid_dataset)
def cifar10_valid_dataset(c: CIFAR10Configs):
    return _cifar_dataset(False, c.dataset_transforms)


@CIFAR10Configs.calc(CIFAR10Configs.train_loader)
def cifar10_train_loader(c: CIFAR10Configs):
    return DataLoader(c.train_dataset,
                      batch_size=c.train_batch_size,
                      shuffle=c.train_loader_shuffle)


@CIFAR10Configs.calc(CIFAR10Configs.valid_loader)
def cifar10_valid_loader(c: CIFAR10Configs):
    return DataLoader(c.valid_dataset,
                      batch_size=c.valid_batch_size,
                      shuffle=c.valid_loader_shuffle)


CIFAR10Configs.aggregate(CIFAR10Configs.dataset_name, 'CIFAR10',
                         (CIFAR10Configs.dataset_transforms, 'cifar10_transforms'),
                         (CIFAR10Configs.train_dataset, 'cifar10_train_dataset'),
                         (CIFAR10Configs.valid_dataset, 'cifar10_valid_dataset'),
                         (CIFAR10Configs.train_loader, 'cifar10_train_loader'),
                         (CIFAR10Configs.valid_loader, 'cifar10_valid_loader'))


class TextDataset:
    itos: List[str]
    stoi: Dict[str, int]
    n_tokens: int
    train: str
    valid: str
    standard_tokens: List[str] = []

    @staticmethod
    def load(path: PurePath):
        with open(str(path), 'r') as f:
            return f.read()

    def __init__(self, path: PurePath, tokenizer: Callable, train: str, valid: str, test: str, *,
                 n_tokens: Optional[int] = None,
                 stoi: Optional[Dict[str, int]] = None,
                 itos: Optional[List[str]] = None):
        self.test = test
        self.valid = valid
        self.train = train
        self.tokenizer = tokenizer
        self.path = path

        if n_tokens or stoi or itos:
            assert stoi and itos and n_tokens
            self.n_tokens = n_tokens
            self.stoi = stoi
            self.itos = itos
        else:
            self.n_tokens = len(self.standard_tokens)
            self.stoi = {t: i for i, t in enumerate(self.standard_tokens)}

            with monit.section("Tokenize"):
                tokens = self.tokenizer(self.train) + self.tokenizer(self.valid)
                tokens = sorted(list(set(tokens)))

            for t in monit.iterate("Build vocabulary", tokens):
                self.stoi[t] = self.n_tokens
                self.n_tokens += 1

            self.itos = [''] * self.n_tokens
            for t, n in self.stoi.items():
                self.itos[n] = t

    def text_to_i(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text)
        return torch.tensor([self.stoi[s] for s in tokens if s in self.stoi], dtype=torch.long)

    def __repr__(self):
        return f'{len(self.train) / 1_000_000 :,.2f}M, {len(self.valid) / 1_000_000 :,.2f}M - {str(self.path)}'


class SequentialDataLoader(IterableDataset):
    def __init__(self, *, text: str, dataset: TextDataset,
                 batch_size: int, seq_len: int):
        self.seq_len = seq_len
        data = dataset.text_to_i(text)
        n_batch = data.shape[0] // batch_size
        data = data.narrow(0, 0, n_batch * batch_size)
        data = data.view(batch_size, -1).t().contiguous()
        self.data = data

    def __len__(self):
        return self.data.shape[0] // self.seq_len

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.data.shape[0] - 1:
            raise StopIteration()

        seq_len = min(self.seq_len, self.data.shape[0] - 1 - self.idx)
        i = self.idx + seq_len
        data = self.data[self.idx: i]
        target = self.data[self.idx + 1: i + 1]
        self.idx = i
        return data, target

    def __getitem__(self, idx):
        seq_len = min(self.seq_len, self.data.shape[0] - 1 - idx)
        i = idx + seq_len
        data = self.data[idx: i]
        target = self.data[idx + 1: i + 1]
        return data, target


class SequentialUnBatchedDataset(Dataset):
    def __init__(self, *, text: str, dataset: TextDataset,
                 seq_len: int,
                 is_random_offset: bool = True):
        self.is_random_offset = is_random_offset
        self.seq_len = seq_len
        self.data = dataset.text_to_i(text)

    def __len__(self):
        return (self.data.shape[0] - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        assert start + self.seq_len + 1 <= self.data.shape[0]
        if self.is_random_offset:
            start += random.randint(0, min(self.seq_len - 1, self.data.shape[0] - (start + self.seq_len + 1)))

        end = start + self.seq_len
        data = self.data[start: end]
        target = self.data[start + 1: end + 1]
        return data, target


class TextFileDataset(TextDataset):
    standard_tokens = []

    def __init__(self, path: PurePath, tokenizer: Callable, *,
                 url: Optional[str] = None,
                 filter_subset: Optional[int] = None):
        path = Path(path)
        if not path.exists():
            if not url:
                raise FileNotFoundError(str(path))
            else:
                download_file(url, path)

        with monit.section("Load data"):
            text = self.load(path)
            if filter_subset:
                text = text[:filter_subset]
            split = int(len(text) * .9)
            train = text[:split]
            valid = text[split:]

        super().__init__(path, tokenizer, train, valid, '')


def _test_tiny_shakespeare():
    from labml import lab
    _ = TextFileDataset(lab.get_data_path() / 'tiny_shakespeare.txt', lambda x: list(x),
                        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')


if __name__ == '__main__':
    _test_tiny_shakespeare()
