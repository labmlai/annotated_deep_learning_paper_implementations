from typing import Tuple

import torch

from labml import experiment, monit
from labml import logger
from labml.logger import Text
from labml_helpers.datasets.text import TextDataset
from labml_nn.sampling import Sampler
from labml_nn.sampling.greedy import GreedySampler
from labml_nn.sampling.nucleus import NucleusSampler
from labml_nn.sampling.temperature import TemperatureSampler
from labml_nn.sampling.top_k import TopKSampler
from labml_nn.transformers.basic.autoregressive_experiment import Configs, AutoregressiveTransformer


def get_model_dataset(run_uuid: str) -> Tuple[AutoregressiveTransformer, TextDataset]:
    experiment.evaluate()

    conf = Configs()

    experiment.configs(conf, experiment.load_configs(run_uuid))

    experiment.load(run_uuid)

    experiment.add_pytorch_models({'model': conf.model})

    experiment.start()

    return conf.model, conf.text


def sample(model, ds, sampler: Sampler, n_samples: int, n_tokens: int, seq_len: int, prompt: str):
    with torch.no_grad():
        data = torch.tile(ds.text_to_i(prompt)[:, None], (1, n_samples))

        # Collect output for printing
        logs = [[(prompt, Text.meta)] for _ in range(n_samples)]
        # Sample 25 tokens
        for i in monit.iterate('Sample', n_tokens):
            # Tokenize the prompt
            data = data[-seq_len:]
            # Get the model output
            logits, *_ = model(data)
            logits = logits[-1]
            # Get the model prediction (greedy)
            res = sampler(logits)
            data = torch.cat([data, res[None, :]], dim=0)
            # Add the prediction for logging
            for j in range(n_samples):
                logs[j] += [('' + ds.itos[res[j]], Text.value)]

    # Print the sampled output
    for j in range(n_samples):
        logger.log(logs[j])


def main():
    model, ds = get_model_dataset('074d4004cc6b11ecad7a0242ac1c0002')
    model.eval()

    with monit.section('greedy'):
        sample(model, ds, GreedySampler(), 4, 32, 128, 'It is')

    with monit.section('temperature=1.'):
        sample(model, ds, TemperatureSampler(1.), 4, 32, 128, 'It is')
    with monit.section('temperature=.1'):
        sample(model, ds, TemperatureSampler(.1), 4, 32, 128, 'It is')
    with monit.section('temperature=10.'):
        sample(model, ds, TemperatureSampler(10.), 4, 32, 128, 'It is')

    with monit.section('top_k=5'):
        sample(model, ds, TopKSampler(2, TemperatureSampler(1.)), 4, 32, 128, 'It is')

    with monit.section('nucles p=.95'):
        sample(model, ds, NucleusSampler(0.95, TemperatureSampler(1.)), 4, 32, 128, 'It is')
    with monit.section('nucles p=.95'):
        sample(model, ds, NucleusSampler(0.1, TemperatureSampler(1.)), 4, 32, 128, 'It is')


if __name__ == '__main__':
    main()
