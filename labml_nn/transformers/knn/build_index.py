"""
---
title: Build FAISS index for k-NN search
summary: This builds the FAISS index with the transformer embeddings.
---

# Build FAISS index for k-NN search

We want to build the index of $\big(f(c_i), w_i\big)$.
We store $f(c_i)$ and $w_i$ in memory mapped numpy arrays.
We find $f(c_i)$ nearest to $f(c_t)$ using [FAISS](https://github.com/facebookresearch/faiss).
FAISS indexes $\big(f(c_i), i\big)$ and we query it with $f(c_t)$.
"""

from typing import Optional

import faiss
import numpy as np
import torch

from labml import experiment, monit, lab
from labml.utils.pytorch import get_modules
from labml_nn.transformers.knn.train_model import Configs


def load_experiment(run_uuid: str, checkpoint: Optional[int] = None):
    """
    Load a saved experiment from [train model](train_model.html).
    """

    # Create configurations object
    conf = Configs()
    # Load custom configurations used in the experiment
    conf_dict = experiment.load_configs(run_uuid)
    # We need to get inputs to the feed forward layer, $f(c_i)$
    conf_dict['is_save_ff_input'] = True

    # This experiment is just an evaluation; i.e. nothing is tracked or saved
    experiment.evaluate()
    # Initialize configurations
    experiment.configs(conf, conf_dict)
    # Set models for saving/loading
    experiment.add_pytorch_models(get_modules(conf))
    # Specify the experiment to load from
    experiment.load(run_uuid, checkpoint)

    # Start the experiment; this is when it actually loads models
    experiment.start()

    return conf


def gather_keys(conf: Configs):
    """
    ## Gather $\big(f(c_i), w_i\big)$ and save them in numpy arrays

    *Note that these numpy arrays will take up a lot of space (even few hundred gigabytes)
    depending on the size of your dataset*.
    """

    # Dimensions of $f(c_i)$
    d_model = conf.transformer.d_model
    # Training data loader
    data_loader = conf.trainer.data_loader
    # Number of contexts; i.e. number of tokens in the training data minus one.
    # $\big(f(c_i), w_i\big)$ for $i \in [2, T]$
    n_keys = data_loader.data.shape[0] * data_loader.data.shape[1] - 1
    # Numpy array for $f(c_i)$
    keys_store = np.memmap(str(lab.get_data_path() / 'keys.npy'), dtype=np.float32, mode='w+', shape=(n_keys, d_model))
    # Numpy array for $w_i$
    vals_store = np.memmap(str(lab.get_data_path() / 'vals.npy'), dtype=np.int, mode='w+', shape=(n_keys, 1))

    # Number of keys $f(c_i)$ collected
    added = 0
    with torch.no_grad():
        # Loop through data
        for i, batch in monit.enum("Collect data", data_loader, is_children_silent=True):
            # $w_i$ the target labels
            vals = batch[1].view(-1, 1)
            # Input data moved to the device of the model
            data = batch[0].to(conf.device)
            # Run the model
            _ = conf.model(data)
            # Get $f(c_i)$
            keys = conf.model.ff_input.view(-1, d_model)
            # Save keys, $f(c_i)$ in the memory mapped numpy array
            keys_store[added: added + keys.shape[0]] = keys.cpu()
            # Save values, $w_i$ in the memory mapped numpy array
            vals_store[added: added + keys.shape[0]] = vals
            # Increment the number of collected keys
            added += keys.shape[0]


def build_index(conf: Configs, n_centeroids: int = 2048, code_size: int = 64, n_probe: int = 8, n_train: int = 200_000):
    """
    ## Build FAISS index

    [Getting started](https://github.com/facebookresearch/faiss/wiki/Getting-started),
    [faster search](https://github.com/facebookresearch/faiss/wiki/Faster-search),
    and [lower memory footprint](https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint)
    tutorials on FAISS will help you learn more about FAISS usage.
    """
    # Dimensions of $f(c_i)$
    d_model = conf.transformer.d_model
    # Training data loader
    data_loader = conf.trainer.data_loader
    # Number of contexts; i.e. number of tokens in the training data minus one.
    # $\big(f(c_i), w_i\big)$ for $i \in [2, T]$
    n_keys = data_loader.data.shape[0] * data_loader.data.shape[1] - 1

    # Build an index with Verenoi cell based faster search with compression that
    # doesn't store full vectors.
    quantizer = faiss.IndexFlatL2(d_model)
    index = faiss.IndexIVFPQ(quantizer, d_model, n_centeroids, code_size, 8)
    index.nprobe = n_probe

    # Load the memory mapped numpy array of keys
    keys_store = np.memmap(str(lab.get_data_path() / 'keys.npy'), dtype=np.float32, mode='r', shape=(n_keys, d_model))

    # Pick a random sample of keys to train the index with
    random_sample = np.random.choice(np.arange(n_keys), size=[min(n_train, n_keys)], replace=False)

    with monit.section('Train index'):
        # Train the index to store the keys
        index.train(keys_store[random_sample])

    # Add keys to the index; $\big(f(c_i), i\big)$
    for s in monit.iterate('Index', range(0, n_keys, 1024)):
        e = min(s + 1024, n_keys)
        # $f(c_i)$
        keys = keys_store[s:e]
        # $i$
        idx = np.arange(s, e)
        # Add to index
        index.add_with_ids(keys, idx)

    with monit.section('Save'):
        # Save the index
        faiss.write_index(index, str(lab.get_data_path() / 'faiss.index'))


def main():
    # Load the experiment. Replace the run uuid with you run uuid from
    # [training the model](train_model.html).
    conf = load_experiment('4984b85c20bf11eb877a69c1a03717cd')
    # Set model to evaluation mode
    conf.model.eval()

    # Collect $\big(f(c_i), w_i\big)$
    gather_keys(conf)
    # Add them to the index for fast search
    build_index(conf)


if __name__ == '__main__':
    main()
