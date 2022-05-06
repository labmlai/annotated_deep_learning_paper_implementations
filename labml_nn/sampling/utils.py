from typing import Tuple

from labml import experiment, monit
from labml import logger
from labml.logger import Text
from labml_helpers.datasets.text import TextDataset

from labml_nn.transformers.basic.autoregressive_experiment import Configs, AutoregressiveTransformer


def get_model_dataset(run_uuid: str) -> Tuple[AutoregressiveTransformer, TextDataset]:
    experiment.evaluate()

    conf = Configs()

    experiment.configs(conf, experiment.load_configs(run_uuid))

    experiment.load(run_uuid)

    experiment.add_pytorch_models({'model': conf.model})

    experiment.start()

    return conf.model, conf.text


def main():
    model, ds = get_model_dataset('074d4004cc6b11ecad7a0242ac1c0002')

    # Starting prompt
    prompt = 'It is'
    # Collect output for printing
    log = [(prompt, Text.subtle)]
    # Sample 25 tokens
    for i in monit.iterate('Sample', 1000):
        # Tokenize the prompt
        data = ds.text_to_i(prompt).unsqueeze(-1)[-128:]
        # Get the model output
        output, *_ = model(data)
        # Get the model prediction (greedy)
        output = output.argmax(dim=-1).squeeze()
        # Add the prediction to prompt
        prompt += '' + ds.itos[output[-1]]
        # Add the prediction for logging
        log += [('' + ds.itos[output[-1]], Text.value)]

    # Print the sampled output
    logger.log(log)


if __name__ == '__main__':
    main()
