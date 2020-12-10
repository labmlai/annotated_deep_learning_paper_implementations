"""
---
title: Evaluate k-nearest neighbor language model
summary: >
  This runs the kNN model and merges the kNN results with transformer output to
  achieve better results than just using the transformer.
---

# Evaluate k-nearest neighbor language model
"""
from typing import Optional, List

import faiss
import numpy as np
import torch

from labml import monit, lab
from labml.logger import inspect
from labml_nn.transformers.knn.train_model import Configs


def knn(queries: torch.Tensor, index: faiss.IndexFlatL2, keys_store: np.ndarray, vals_store: np.ndarray, n_tokens: int):
    """
    ## $k$-NN to get $p(w_t, c_t)$

    Here we refer to $f($\color{yellowgreen}{c_t})$ as queries,
    $f(c_i)$ as keys and $w_i$ as values.
    """

    # Save shape of queries to reshape results
    queries_shape = queries.shape

    # Flatten the `batch` and `sequence` dimensions of queries
    queries = queries.view(-1, queries_shape[-1])

    # Find 10 nearest neighbors of $f($\color{yellowgreen}{c_t})$ among $f(c_i)$.
    # `distance` is the distance given by FAISS and `idx`, $i$ is the index of it in `keys_store`.
    distance, idx = index.search(queries.numpy(), 10)

    # Get $f(c_i)$
    keys_found = queries.new_tensor(keys_store[idx])
    # Get $w_i$
    vals_found = torch.tensor(vals_store[idx]).squeeze(-1)

    # We are going to calculate the cosine similarity between normalized vectors

    # Normalize $f(c_i)$
    keys_found_n = keys_found / torch.sqrt((keys_found ** 2).sum(-1, keepdims=True) + 1e-10)
    # Normalize $f($\color{yellowgreen}{c_t})$
    queries_n = queries / torch.sqrt((queries ** 2).sum(-1, keepdims=True) + 1e-10)

    # Get the dot-product, or cosine similarity
    dot_prod = (keys_found_n * queries_n.unsqueeze(1)).sum(-1)

    # Token-wise logits
    logits_token = dot_prod.new_zeros(queries.shape[0], n_tokens)
    # Scatter and accumulate token logits based on the nearest neighbors
    _ = logits_token.scatter_(dim=1, index=vals_found, src=dot_prod, reduce='add')

    # Reshape the logits
    logits_token = logits_token.reshape(queries_shape[0], queries_shape[1], -1)

    return logits_token


def validation_loss(knn_weights: List[float], last_n: Optional[int], conf: Configs, index: faiss.IndexFlatL2,
                    keys_store: np.ndarray, vals_store: np.ndarray):
    """
    ## Calculate validation loss

    We calculate the validation loss of the combined on $k$-NN prediction and transformer prediction.
    The weight given to the $k$-NN model is given by `knn_weight`.
    It's a list of weights and we calculate the validation loss for each.
    """

    # List of losses for each `knn_weights`
    losses = [[] for _ in knn_weights]
    # Number of samples in each batch
    n_samples = []
    with torch.no_grad():
        # Iterate through validation data
        for i, batch in monit.enum("Validation", conf.validator.data_loader, is_children_silent=True):
            # Get data and target labels
            data, target = batch[0].to(conf.device), batch[1].to(conf.device)
            # Run the model and get predictions $p(w_t, c_t)$
            res = conf.model(data)
            # Get $k$-NN predictions
            res_knn = knn(conf.model.ff_input.cpu(), index, keys_store, vals_store, conf.n_tokens)
            res_knn = res_knn.to(conf.device)

            # This is to calculate only the loss for `last_n` tokens.
            # This is important because the first predictions (along the sequence)
            # of transformer model has very few past tokens to look at.
            if last_n:
                res = res[-last_n:]
                res_knn = res_knn[-last_n:]
                target = target[-last_n:]

            # Number of samples
            n_s = res.shape[0] * data.shape[1]
            n_samples.append(n_s)

            # Calculate scores for each of `knn_weights`.
            for i, c in enumerate(knn_weights):
                # Calculate the loss
                loss = conf.loss_func(res_knn * c + (1 - c) * res, target)
                losses[i].append(loss * n_s)

    return losses, n_samples


def load_index(conf: Configs, n_probe: int = 8):
    """
    ## Load the index
    """
    # Dimensions of $f(c_i)$
    d_model = conf.transformer.d_model
    # Training data loader
    data_loader = conf.trainer.data_loader
    # Number of contexts; i.e. number of tokens in the training data minus one.
    # $\big(f(c_i), w_i\big)$ for $i \in [2, T]$
    n_keys = data_loader.data.shape[0] * data_loader.data.shape[1] - 1

    # Load FAISS index
    with monit.section('Load index'):
        index = faiss.read_index(str(lab.get_data_path() / 'faiss.index'))
    # Set number of cells to probe
    index.nprobe = n_probe

    # Load memory mapped numpy arrays
    keys_store = np.memmap(str(lab.get_data_path() / 'keys.npy'), dtype=np.float32, mode='r', shape=(n_keys, d_model))
    vals_store = np.memmap(str(lab.get_data_path() / 'vals.npy'), dtype=np.int, mode='r', shape=(n_keys, 1))

    return index, keys_store, vals_store


def main():
    from labml_nn.transformers.knn.build_index import load_experiment
    # Load the experiment. Replace the run uuid with you run uuid from
    # [training the model](train_model.html).
    conf = load_experiment('4984b85c20bf11eb877a69c1a03717cd')
    # Set model to evaluation mode
    conf.model.eval()

    # Load index
    index, keys_store, vals_store = load_index(conf)
    # List of weights given to $k$-NN prediction. We will evaluate the validation loss for
    # each of the weights
    knn_weights = [i / 20 for i in range(10)]
    # Evaluate validation loss
    losses, n_samples = validation_loss(knn_weights, None, conf, index, keys_store, vals_store)
    # Output the losses for each of `knn_weights`.
    inspect({c: np.sum(losses[i]) / np.sum(n_samples) for i, c in enumerate(knn_weights)})


if __name__ == '__main__':
    main()
