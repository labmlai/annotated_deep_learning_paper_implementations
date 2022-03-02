import json
from pathlib import Path
from typing import List, Callable

import faiss
import numpy as np
import torch
from torch.utils.data import Dataset as PyTorchDataset

from labml import lab, monit
from labml_helpers.datasets.text import TextFileDataset, TextDataset
from labml_nn.transformers.retro.bert_embeddings import BERTChunkEmbeddings


class Index:
    def __init__(self, n_probe: int = 8,
                 n_extra: int = 4, n_neighbors: int = 4,
                 exclude_neighbor_span: int = 64, chunk_length: int = 64):
        self.n_neighbors = n_neighbors
        self.chunk_length = chunk_length
        self.exclude_neighbor_span = exclude_neighbor_span
        self.n_extra = n_extra

        self.bert = BERTChunkEmbeddings(torch.device('cuda:0'))
        with monit.section('Load index'):
            self.index = faiss.read_index(str(lab.get_data_path() / 'retro.index'))
        self.index.nprobe = n_probe

    def filter_neighbors(self, offset: int, neighbor_offsets: List[int]):
        return [n for n in neighbor_offsets
                if n < offset - (self.chunk_length + self.exclude_neighbor_span)
                or n > offset + (self.chunk_length + self.exclude_neighbor_span)]

    def __call__(self, chunks: List[str], offsets: List[int]):
        emb = self.bert(chunks).cpu()

        distance, neighbor_offsets = self.index.search(emb.numpy(), self.n_neighbors + self.n_extra)

        neighbor_offsets = [self.filter_neighbors(off, n_off)[:self.n_neighbors] for off, n_off in
                            zip(offsets, neighbor_offsets)]

        return neighbor_offsets


def build_database(chunk_length: int = 64, chunks_per_sample: int = 8, offset_noise: int = 8):
    dataset = TextFileDataset(
        lab.get_data_path() / 'tiny_shakespeare.txt',
        list,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

    text = dataset.train

    index = Index()
    sample_offsets = []
    i = 0
    while i < len(text):
        skip = np.random.randint(offset_noise)
        i += skip
        if i >= len(text):
            break

        sample_offsets.append(i)

        i += chunks_per_sample * chunk_length

    samples = []
    for i in monit.iterate('Gather Neighbors', sample_offsets):
        sample = text[i: i + chunks_per_sample * chunk_length + 1]
        src = sample[:-1]
        chunks = [src[j:j + chunk_length] for j in range(0, len(src), chunk_length)]
        chunk_offsets = [j for j in range(0, len(src), chunk_length)]

        neighbor_offsets = index(chunks, chunk_offsets)

        neighbors = [[text[j: j + chunk_length * 2] for j in n_off] for n_off in neighbor_offsets]

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
