"""
---
title: Text Dataset for GPT-NeoX
summary: >
    Loads text datasets to fine-tune GPT-NeoX
---

# Text Dataset for GPT-NeoX
"""
from pathlib import PurePath, Path
from typing import Optional, List

import torch
import torch.utils.data
from labml import lab
from labml import monit
from labml.logger import inspect
from labml.utils.download import download_file

from labml_nn.neox.tokenizer import get_tokenizer


def load_text(path: PurePath, url: Optional[str] = None, *, filter_subset: Optional[int] = None):
    """
    ### Load text file

    :param path: is the location of the text file
    :param url: is the URL to download the file from
    :param filter_subset: is the number of characters to filter.
     Use this during testing when trying large datasets
    :return: the text content
    """

    path = Path(path)

    # Download if it doesn't exist
    if not path.exists():
        if not url:
            raise FileNotFoundError(str(path))
        else:
            download_file(url, path)

    with monit.section("Load data"):
        # Load data
        with open(str(path), 'r') as f:
            text = f.read()
        # Filter
        if filter_subset:
            text = text[:filter_subset]

    #
    return text


class NeoXDataset(torch.utils.data.Dataset):
    """
    ## Dataset for fine-tuning GPT-NeoX

    This is not optimized to very large datasets.
    """

    def __init__(self, tokens: List[int], seq_len: int):
        """
        :param tokens: is the list of token ids
        :param seq_len: is the sequence length of a single training sample
        """

        self.seq_len = seq_len
        # Number of samples
        n_samples = len(tokens) // seq_len
        self.n_samples = n_samples
        # Truncate
        tokens = tokens[:n_samples * seq_len + 1]
        # Create a PyTorch tensor
        self.tokens = torch.tensor(tokens)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        """
        ### Get a sample

        :param idx: is the index of the sample
        :return: the input and the target
        """
        offset = idx * self.seq_len
        return self.tokens[offset:offset + self.seq_len], self.tokens[offset + 1:offset + 1 + self.seq_len]


DATASETS = {
    'tiny_shakespeare': {
        'file': 'tiny_shakespeare.txt',
        'url': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    }
}


def get_training_data(seq_len: int = 32, dataset_name: str = 'tiny_shakespeare', truncate: int = -1):
    """
    ### Load Dataset

    :param seq_len: is the sequence length of a single training sample
    :param dataset_name: is the name of the dataset
    :return: the dataset
    """

    ds = DATASETS[dataset_name]
    # Load the content
    text = load_text(lab.get_data_path() / ds['file'], ds['url'])
    # Tokenize
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode_batch([text])[0]

    if truncate > 0:
        token_ids = tokens.ids[:truncate * seq_len]
    else:
        token_ids = tokens.ids

    #
    return NeoXDataset(token_ids, seq_len)


def _test():
    dataset = get_training_data()

    inspect(tokens=len(dataset.tokens))


#
if __name__ == '__main__':
    _test()
