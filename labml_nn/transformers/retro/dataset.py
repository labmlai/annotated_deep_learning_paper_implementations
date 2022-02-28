import faiss
import numpy as np
import torch
from labml.logger import inspect

from labml import lab, monit
from labml_helpers.datasets.text import TextFileDataset
from labml_nn.transformers.retro.bert_embeddings import BERTChunkEmbeddings


def build_database(chunk_length: int=64):
    dataset = TextFileDataset(
        lab.get_data_path() / 'tiny_shakespeare.txt',
        list,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

    text = dataset.train

    skipped_text = []
    i = 0
    while i < len(text):
        skip = np.random.randint(64)
        i += skip
        if i >= len(text):
            break

        split = text[i: i * 64 * chunk_length + 1]

        chunk_src = [split[j:j + chunk_length] for j in range(0, len(text), chunk_length)]
        chunk_offsets = [j for j in range(0, len(text), chunk_length)]
        chunk_tgt = [split[i:i + chunk_length] for i in range(1, len(text), chunk_length)]

        neighbors = get_neighbors(chunk_src, chunk_offsets)

        i += 64 * chunk_length




if __name__ == '__main__':
    build_database()
