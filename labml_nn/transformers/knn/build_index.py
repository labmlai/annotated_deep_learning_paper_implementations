from typing import Optional

import faiss
import numpy as np
import torch

from labml import experiment, monit, lab
from labml.utils.pytorch import get_modules
from labml_nn.transformers.knn.train_model import Configs


def load_experiment(run_uuid: str, checkpoint: Optional[int] = None):
    conf = Configs()
    conf_dict = experiment.load_configs(run_uuid)
    conf_dict['is_save_ff_input'] = True

    experiment.evaluate()
    experiment.configs(conf, conf_dict, 'run')
    experiment.add_pytorch_models(get_modules(conf))
    experiment.load(run_uuid, checkpoint)

    experiment.start()

    return conf


def gather_keys(conf: Configs):
    d_model = conf.transformer.d_model
    data_loader = conf.trainer.data_loader
    n_keys = data_loader.data.shape[0] * data_loader.data.shape[1] - 1
    keys_store = np.memmap(str(lab.get_data_path() / 'keys.npy'), dtype=np.float32, mode='w+', shape=(n_keys, d_model))
    vals_store = np.memmap(str(lab.get_data_path() / 'vals.npy'), dtype=np.int, mode='w+', shape=(n_keys, 1))

    added = 0
    with torch.no_grad():
        for i, batch in monit.enum("Collect data", data_loader, is_children_silent=True):
            vals = batch[1].view(-1, 1)
            data = batch[0].to(conf.device)
            _ = conf.model(data)
            keys = conf.model.ff_input.view(-1, d_model)
            keys = keys  # / torch.sqrt((keys ** 2).sum(-1, keepdims=True) + 1e-10)
            keys_store[added: added + keys.shape[0]] = keys.cpu()
            vals_store[added: added + keys.shape[0]] = vals
            added += keys.shape[0]


def build_index(conf: Configs, n_centeroids: int = 2048, code_size: int = 64, n_probe: int = 8, n_train: int = 200_000):
    d_model = conf.transformer.d_model
    data_loader = conf.trainer.data_loader
    n_keys = data_loader.data.shape[0] * data_loader.data.shape[1] - 1

    quantizer = faiss.IndexFlatL2(d_model)
    index = faiss.IndexIVFPQ(quantizer, d_model, n_centeroids, code_size, 8)
    index.nprobe = n_probe

    keys_store = np.memmap(str(lab.get_data_path() / 'keys.npy'), dtype=np.float32, mode='r', shape=(n_keys, d_model))

    random_sample = np.random.choice(np.arange(n_keys), size=[min(n_train, n_keys)], replace=False)

    with monit.section('Train index'):
        index.train(keys_store[random_sample])

    for s in monit.iterate('Index', range(0, n_keys, 1024)):
        e = min(s + 1024, n_keys)
        keys = keys_store[s:e]
        idx = np.arange(s, e)
        index.add_with_ids(keys, idx)

    with monit.section('Save'):
        faiss.write_index(index, str(lab.get_data_path() / 'faiss.index'))


def main():
    conf = load_experiment('4984b85c20bf11eb877a69c1a03717cd')
    conf.model.eval()

    gather_keys(conf)
    build_index(conf)


if __name__ == '__main__':
    main()
