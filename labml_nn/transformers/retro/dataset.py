import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as PyTorchDataset

from labml import lab, monit
from labml_helpers.datasets.text import TextFileDataset, TextDataset
from labml_nn.transformers.retro.database import RetroIndex


def build_database(chunk_len: int = 16, chunks_per_sample: int = 32, offset_noise: int = 8):
    dataset = TextFileDataset(
        lab.get_data_path() / 'tiny_shakespeare.txt',
        list,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

    text = dataset.train

    index = RetroIndex()
    sample_offsets = []
    i = 0
    while i < len(text):
        skip = np.random.randint(offset_noise)
        i += skip
        if i + chunks_per_sample * chunk_len > len(text):
            break

        sample_offsets.append(i)

        i += chunks_per_sample * chunk_len

    samples = []
    for i in monit.iterate('Gather Neighbors', sample_offsets):
        sample = text[i: i + chunks_per_sample * chunk_len + 1]
        src = sample[:-1]
        chunks = [src[j:j + chunk_len] for j in range(0, len(src), chunk_len)]
        chunk_offsets = [j + i for j in range(0, len(src), chunk_len)]

        neighbor_offsets = index(chunks, chunk_offsets)

        # for n_off, offset in zip(neighbor_offsets, chunk_offsets):
        #     n_off[-1] = offset

        neighbors = [[text[j: j + chunk_len * 2] for j in n_off] for n_off in neighbor_offsets]

        samples.append((sample[:-1], sample[1:], neighbors))

    with open(str(lab.get_data_path() / 'retro_train_dataset.json'), 'w') as f:
        f.write(json.dumps(samples))


class Dataset(PyTorchDataset):
    def __init__(self, file_path: Path, tds: TextDataset = list):
        self.tds = tds
        with open(str(file_path), 'r') as f:
            self.samples = json.loads(f.read())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        src = self.tds.text_to_i(s[0])
        tgt = self.tds.text_to_i(s[1])
        neighbors = torch.stack([torch.stack([self.tds.text_to_i(n) for n in chunks]) for chunks in s[2]])
        return src, tgt, neighbors


if __name__ == '__main__':
    build_database()
