"""
---
title: Database for nearest neighbor retrieval
summary: >
  Nearest neighbor retrieval and creation of the database
---

# Database for nearest neighbor retrieval

This is the build the database and retrieves nearest neighbors for
 [RETRO model](index.html).

We use [FAISS library](https://faiss.ai/) for the database whilst the paper had used the SCaNN library.
"""

from typing import List, Optional

import faiss
import numpy as np
import torch

from labml import lab, monit
from labml_helpers.datasets.text import TextFileDataset
from labml_nn.transformers.retro.bert_embeddings import BERTChunkEmbeddings


def build_database(chunk_len: int = 16, batch_size: int = 64, d_emb: int = 768, n_centeroids: int = 256,
                   code_size: int = 64, n_probe: int = 8, n_train: int = 50_000):
    """
    ## Build Database

    * `chunk_len` is the length of a chunk (number of characters)
    * `batch_size` is the batch size to use when calculating $\text{B\small{ERT}}(N)$
    * `d_emb` is the number of features in $\text{B\small{ERT}}(N)$ embeddings
        [lists to select in FAISS index](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVFPQ.html)
    * `n_centeroids` is the number of lists in the index
    * `code_size` encoded vector size in the index
    * `n_probe` is the number of lists to probe
    * `n_train' is the number of keys to train the index on
    """

    # Load the dataset text file
    dataset = TextFileDataset(
        lab.get_data_path() / 'tiny_shakespeare.txt',
        list,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

    # Get training data (a string)
    text = dataset.train

    # Split the text into chunks of `chunk_length`
    chunks = [text[i:i + chunk_len] for i in range(0, len(text), chunk_len) if i + chunk_len * 2 < len(text)]
    # Get the offsets of each of the chunks
    chunk_offsets = np.array([i for i in range(0, len(text), chunk_len) if i + chunk_len * 2 < len(text)])
    # Number of chunks
    n_chunks = len(chunks)

    # Initialize BERT to get $\text{B\small{ERT}}(N)$
    bert = BERTChunkEmbeddings(torch.device('cuda:0'))

    # Get chunk embeddings by processing `batch_size` number of chunks on each iteration
    chunk_emb = []
    for i in monit.iterate('Get embeddings', range(0, n_chunks, batch_size)):
        chunk_emb.append(bert(chunks[i: i + batch_size]).cpu())
    # Merge them into a single tensor
    chunk_emb = torch.cat(chunk_emb, dim=0).numpy()

    # Create the [FAISS index](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVFPQ.html)
    quantizer = faiss.IndexFlatL2(d_emb)
    index = faiss.IndexIVFPQ(quantizer, d_emb, n_centeroids, code_size, 8)
    index.nprobe = n_probe

    # Get a random sample of the the chunk indexes
    random_sample = np.random.choice(np.arange(n_chunks), size=[min(n_train, n_chunks)], replace=False)

    # Train the index to store the keys
    with monit.section('Train index'):
        index.train(chunk_emb[random_sample])

    # Add the chunks to the index in batches of size `1024`
    for s in monit.iterate('Index', range(0, n_chunks, 1024)):
        e = min(s + 1024, n_chunks)
        # Add to index
        index.add_with_ids(chunk_emb[s:e], chunk_offsets[s: e])

    # Save the index
    with monit.section('Save'):
        faiss.write_index(index, str(lab.get_data_path() / 'retro.index'))


class RetroIndex:
    """
    ## Index for retrieving nearest neighbors
    """

    def __init__(self, chunk_len: int = 16, n_probe: int = 8,
                 n_neighbors: int = 2, n_extra: int = 2,
                 exclude_neighbor_span: int = 8):
        """
        * `chunk_len` is the chunk length
        * `n_probe` is the number of lists to probe
        * `n_neighbors` is the number of neighbors to retrieve
        * `n_extra` is the number of extra neighbors to retrieve since we will be
            removing neighbors overlapping with the query chunk
        * `exclude_neighbor_span` is the extra text length to avoid when checking for overlaps
        """

        self.n_neighbors = n_neighbors
        self.chunk_len = chunk_len
        self.exclude_neighbor_span = exclude_neighbor_span
        self.n_extra = n_extra

        # Initialize BERT to get $\text{B\small{ERT}}(N)$
        self.bert = BERTChunkEmbeddings(torch.device('cuda:0'))
        # Load the database
        with monit.section('Load index'):
            self.index = faiss.read_index(str(lab.get_data_path() / 'retro.index'))
            self.index.nprobe = n_probe

    def filter_neighbors(self, offset: int, neighbor_offsets: List[int]):
        """
        #### Filter neighbors that overlap with the query
        
        The positions of the neighbors are given by `neighbor_offsets` and the position
        of the query chunk is `offset`.
        """
        return [n for n in neighbor_offsets
                if n < offset - (self.chunk_len + self.exclude_neighbor_span)
                or n > offset + (self.chunk_len + self.exclude_neighbor_span)]

    def __call__(self, query_chunks: List[str], offsets: Optional[List[int]]):
        """
        ### Retrieve nearest neighbors
        """

        # Get $\text{B\small{ERT}}(N)$ of query chunks
        emb = self.bert(query_chunks).cpu()

        # Get `n_neighbors + n_extra` nearest neighbors from the database
        distance, neighbor_offsets = self.index.search(emb.numpy(), self.n_neighbors + self.n_extra)

        # If the query chunk offsets are given filter out overlapping chunks
        if offsets is not None:
            neighbor_offsets = [self.filter_neighbors(off, n_off)
                                for off, n_off in zip(offsets, neighbor_offsets)]

        # Get the closest `n_neighbors` after filtering
        neighbor_offsets = [n_off[:self.n_neighbors] for n_off in neighbor_offsets]

        #
        return neighbor_offsets


#
if __name__ == '__main__':
    build_database()
