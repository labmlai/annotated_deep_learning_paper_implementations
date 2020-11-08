from typing import Optional, List

import faiss
import numpy as np
import torch
from torch.nn import functional as F

from labml import monit, tracker, lab
from labml.logger import inspect
from labml_nn.transformers.knn.train_model import Configs


def knn(queries: torch.Tensor, index: faiss.IndexFlatL2, keys_store: np.ndarray, vals_store: np.ndarray, n_tokens: int):
    queries_shape = queries.shape
    queries = queries.view(-1, queries_shape[-1])

    distance, idx = index.search(queries.numpy(), 10)

    keys_found = queries.new_tensor(keys_store[idx])
    vals_found = torch.tensor(vals_store[idx]).squeeze(-1)

    keys_found_n = keys_found / torch.sqrt((keys_found ** 2).sum(-1, keepdims=True) + 1e-10)
    queries_n = queries / torch.sqrt((queries ** 2).sum(-1, keepdims=True) + 1e-10)

    dot_prod = (keys_found_n * queries_n.unsqueeze(1)).sum(-1)

    logits_token = dot_prod.new_zeros(queries.shape[0], n_tokens)
    _ = logits_token.scatter_(dim=1, index=vals_found, src=dot_prod, reduce='add')

    logits_token = logits_token.reshape(queries_shape[0], queries_shape[1], -1)

    return logits_token


def validation_loss(coef: List[float], last_n: Optional[int], conf: Configs, index: faiss.IndexFlatL2,
                    keys_store: np.ndarray, vals_store: np.ndarray):
    valid = conf.validator

    losses = [[] for _ in coef]
    samples = []
    with torch.no_grad():
        with tracker.namespace('valid'):
            for i, batch in monit.enum("Validation", valid.data_loader, is_children_silent=True):
                data, tgt = batch[0].to(conf.device), batch[1].to(conf.device)
                res = conf.model(data)
                res_knn = knn(conf.model.ff_input.cpu(), index, keys_store, vals_store, conf.n_tokens)
                res_knn = res_knn.to(conf.device)
                if last_n:
                    res = res[-last_n:]
                    res_knn = res_knn[-last_n:]
                    tgt = tgt[-last_n:]
                s = res.shape[0] * data.shape[1]
                samples.append(s)
                for i, c in enumerate(coef):
                    loss = conf.loss_func(res_knn * c + (1 - c) * res, tgt)
                    losses[i].append(loss * s)
    inspect({c: np.sum(losses[i]) / np.sum(samples) for i, c in enumerate(coef)})

    return losses, samples


def load_index(conf: Configs, n_probe: int = 8):
    d_model = conf.transformer.d_model
    data_loader = conf.trainer.data_loader
    n_keys = data_loader.data.shape[0] * data_loader.data.shape[1] - 1
    with monit.section('Load index'):
        index = faiss.read_index(str(lab.get_data_path() / 'faiss.index'))
    index.nprobe = n_probe

    keys_store = np.memmap(str(lab.get_data_path() / 'keys.npy'), dtype=np.float32, mode='r', shape=(n_keys, d_model))
    vals_store = np.memmap(str(lab.get_data_path() / 'vals.npy'), dtype=np.int, mode='r', shape=(n_keys, 1))

    return index, keys_store, vals_store


def main():
    from labml_nn.transformers.knn.build_index import load_experiment
    conf = load_experiment('4984b85c20bf11eb877a69c1a03717cd')
    conf.model.eval()

    index, keys_store, vals_store = load_index(conf)
    validation_loss([i / 20 for i in range(10)], None, conf, index, keys_store, vals_store)


if __name__ == '__main__':
    main()
