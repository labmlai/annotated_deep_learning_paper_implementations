from typing import Dict

import numpy as np
import torch
from torch import nn

from labml import lab, monit, tracker, experiment
from labml.configs import BaseConfigs
from labml.utils import download
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module
from labml_nn.graphs.gat import GraphAttentionLayer
from labml_nn.optimizers.configs import OptimizerConfigs


class CoraDataset:
    labels: torch.Tensor
    classes: Dict[str, int]
    features: torch.Tensor
    adj_mat: torch.Tensor

    @staticmethod
    def _download():
        # From https://linqs.soe.ucsc.edu/data
        if not (lab.get_data_path() / 'cora').exists():
            download.download_file('https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz',
                                   lab.get_data_path() / 'cora.tgz')
            download.extract_tar(lab.get_data_path() / 'cora.tgz', lab.get_data_path())

    def __init__(self, include_edges: bool = True):
        self.include_edges = include_edges

        self._download()

        with monit.section('Read content file'):
            content = np.genfromtxt(str(lab.get_data_path() / 'cora/cora.content'), dtype=np.dtype(str))
        with monit.section('Read citations file'):
            citations = np.genfromtxt(str(lab.get_data_path() / 'cora/cora.cites'), dtype=np.int32)

        features = torch.tensor(np.array(content[:, 1:-1], dtype=np.float32))
        self.features = features / features.sum(dim=1, keepdim=True)

        self.classes = {s: i for i, s in enumerate(set(content[:, -1]))}
        self.labels = torch.tensor([self.classes[i] for i in content[:, -1]], dtype=torch.long)

        paper_ids = np.array(content[:, 0], dtype=np.int32)
        ids_to_idx = {id_: i for i, id_ in enumerate(paper_ids)}

        self.adj_mat = torch.eye(len(self.labels), dtype=torch.bool)
        if self.include_edges:
            for e in citations:
                e1, e2 = ids_to_idx[e[0]], ids_to_idx[e[1]]
                self.adj_mat[e1][e2] = True
                self.adj_mat[e2][e1] = True


class GAT(Module):
    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):
        super().__init__()

        self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.output = GraphAttentionLayer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout)

        self.activation = nn.ELU()

    def __call__(self, x: torch.Tensor, adj_mat: torch.Tensor):
        x = self.dropout(x)
        x = self.layer1(x, adj_mat)
        x = self.activation(x)
        x = self.dropout(x)
        return self.output(x, adj_mat)


def accuracy(output: torch.Tensor, labels: torch.Tensor):
    return output.argmax(dim=-1).eq(labels).sum().item() / len(labels)


class Configs(BaseConfigs):
    model: GAT
    training_samples: int = 500
    in_features: int
    n_hidden: int = 64
    n_heads: int = 8
    n_classes: int
    n_heads: int
    dropout: float = 0.6
    include_edges: bool = True
    cora_dataset: CoraDataset
    epochs: int = 1_000
    loss_func = nn.CrossEntropyLoss()

    device: torch.device = DeviceConfigs()
    optimizer: torch.optim.Adam

    def initialize(self):
        self.cora_dataset = CoraDataset(self.include_edges)
        self.n_classes = len(self.cora_dataset.classes)
        self.in_features = self.cora_dataset.features.shape[1]
        self.model = GAT(self.in_features, self.n_hidden, self.n_classes, self.n_heads, self.dropout).to(self.device)
        optimizer_conf = OptimizerConfigs()
        optimizer_conf.parameters = self.model.parameters()
        self.optimizer = optimizer_conf

    def run(self):
        features = self.cora_dataset.features.to(self.device)
        labels = self.cora_dataset.labels.to(self.device)
        edges_adj = self.cora_dataset.adj_mat.to(self.device)
        edges_adj = edges_adj.unsqueeze(-1)

        idx_rand = torch.randperm(len(labels))
        idx_train = idx_rand[:self.training_samples]
        idx_valid = idx_rand[self.training_samples:]

        for epoch in monit.loop(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(features, edges_adj)
            loss = self.loss_func(output[idx_train], labels[idx_train])
            tracker.add('loss.train', loss)
            tracker.add('accuracy.train', accuracy(output[idx_train], labels[idx_train]))
            loss.backward()
            self.optimizer.step()

            self.model.eval()

            with torch.no_grad():
                output = self.model(features, edges_adj)

                loss = self.loss_func(output[idx_valid], labels[idx_valid])
                tracker.add('loss.valid', loss)
                tracker.add('accuracy.valid', accuracy(output[idx_valid], labels[idx_valid]))

            tracker.save()


def main():
    # Create configurations
    conf = Configs()
    # Create an experiment
    experiment.create(name='gat')
    # Calculate configurations.
    # It will calculate `conf.run` and all other configs required by it.
    experiment.configs(conf, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 5e-3,
        'optimizer.weight_decay': 5e-4,
    })
    conf.initialize()

    # Start and watch the experiment
    with experiment.start():
        # Run the training
        conf.run()


if __name__ == '__main__':
    main()
