from typing import List, Optional

import faiss
import numpy as np
import torch

from labml import lab, monit
from labml_helpers.datasets.text import TextFileDataset
from labml_nn.transformers.retro.bert_embeddings import BERTChunkEmbeddings


class RetroIndex:
    def __init__(self, n_probe: int = 8,
                 n_extra: int = 4, n_neighbors: int = 2,
                 exclude_neighbor_span: int = 8, chunk_len: int = 16):
        self.n_neighbors = n_neighbors
        self.chunk_len = chunk_len
        self.exclude_neighbor_span = exclude_neighbor_span
        self.n_extra = n_extra

        self.bert = BERTChunkEmbeddings(torch.device('cuda:0'))
        with monit.section('Load index'):
            self.index = faiss.read_index(str(lab.get_data_path() / 'retro.index'))
        self.index.nprobe = n_probe

    def filter_neighbors(self, offset: int, neighbor_offsets: List[int]):
        # return neighbor_offsets
        return [n for n in neighbor_offsets
                if n < offset - (self.chunk_len + self.exclude_neighbor_span)
                or n > offset + (self.chunk_len + self.exclude_neighbor_span)]

    def __call__(self, chunks: List[str], offsets: Optional[List[int]]):
        emb = self.bert(chunks).cpu()

        distance, neighbor_offsets = self.index.search(emb.numpy(), self.n_neighbors + self.n_extra)

        if offsets is not None:
            neighbor_offsets = [self.filter_neighbors(off, n_off)[:self.n_neighbors] for off, n_off in
                                zip(offsets, neighbor_offsets)]
        else:
            neighbor_offsets = [n_off[:self.n_neighbors] for n_off in neighbor_offsets]

        return neighbor_offsets


def build_database(chunk_len: int = 16, batch_size: int = 64, d_emb: int = 768, n_centeroids: int = 256,
                   code_size: int = 64, n_probe: int = 8, n_train: int = 50_000):
    dataset = TextFileDataset(
        lab.get_data_path() / 'tiny_shakespeare.txt',
        list,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

    text = dataset.train

    chunks = [text[i:i + chunk_len] for i in range(0, len(text), chunk_len) if i + chunk_len * 2 < len(text)]
    chunk_offsets = np.array([i for i in range(0, len(text), chunk_len) if i + chunk_len * 2 < len(text)])
    n_chunks = len(chunks)

    bert = BERTChunkEmbeddings(torch.device('cuda:0'))

    chunk_emb = []

    for i in monit.iterate('Get embeddings', range(0, n_chunks, batch_size)):
        chunk_emb.append(bert(chunks[i: i + batch_size]).cpu())

    chunk_emb = torch.cat(chunk_emb, dim=0).numpy()
    quantizer = faiss.IndexFlatL2(d_emb)
    index = faiss.IndexIVFPQ(quantizer, d_emb, n_centeroids, code_size, 8)
    index.nprobe = n_probe

    random_sample = np.random.choice(np.arange(n_chunks), size=[min(n_train, n_chunks)], replace=False)

    with monit.section('Train index'):
        # Train the index to store the keys
        index.train(chunk_emb[random_sample])

    for s in monit.iterate('Index', range(0, n_chunks, 1024)):
        e = min(s + 1024, n_chunks)
        emb = chunk_emb[s:e]
        # Add to index
        index.add_with_ids(chunk_emb[s:e], chunk_offsets[s: e])

    with monit.section('Save'):
        # Save the index
        faiss.write_index(index, str(lab.get_data_path() / 'retro.index'))


if __name__ == '__main__':
    build_database()
