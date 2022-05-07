import torch

from labml import monit, logger, lab

from labml.logger import Text

from labml_nn.sampling import Sampler
from labml_nn.sampling.greedy import GreedySampler
from labml_nn.sampling.nucleus import NucleusSampler
from labml_nn.sampling.temperature import TemperatureSampler
from labml_nn.sampling.top_k import TopKSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def sample(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, sampler: Sampler,
           n_samples: int, n_tokens: int, seq_len: int, prompt: str):
    with torch.no_grad():
        data = torch.tile(torch.tensor(tokenizer.encode(prompt))[None, :], (n_samples, 1))

        # Collect output for printing
        logs = [[(prompt, Text.meta)] for _ in range(n_samples)]
        # Sample 25 tokens
        for i in monit.iterate('Sample', n_tokens):
            # Tokenize the prompt
            data = data[-seq_len:]
            # Get the model output
            logits = model(data)[0]
            logits = logits[:, -1]
            # Get the model prediction (greedy)
            res = sampler(logits)
            data = torch.cat([data, res[:, None]], dim=1)
            # Add the prediction for logging
            for j in range(n_samples):
                logs[j] += [('' + tokenizer.decode(res[j]), Text.value)]

    # Print the sampled output
    for j in range(n_samples):
        logger.log(logs[j])


def main():
    with monit.section('Load tokenizer/model'):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=lab.get_data_path() / 'cache')
        model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=lab.get_data_path() / 'cache')
    model.eval()

    prompt = 'I saw an interesting dream last night. '

    with monit.section('greedy'):
        sample(model, tokenizer, GreedySampler(), 4, 32, 128, prompt)

    with monit.section('temperature=1.'):
        sample(model, tokenizer, TemperatureSampler(1.), 4, 32, 128, prompt)
    with monit.section('temperature=.1'):
        sample(model, tokenizer, TemperatureSampler(.1), 4, 32, 128, prompt)
    with monit.section('temperature=10.'):
        sample(model, tokenizer, TemperatureSampler(10.), 4, 32, 128, prompt)

    with monit.section('top_k=5'):
        sample(model, tokenizer, TopKSampler(2, TemperatureSampler(1.)), 4, 32, 128, prompt)

    with monit.section('nucleus p=.95'):
        sample(model, tokenizer, NucleusSampler(0.95, TemperatureSampler(1.)), 4, 32, 128, prompt)
    with monit.section('nucleus p=.1'):
        sample(model, tokenizer, NucleusSampler(0.1, TemperatureSampler(1.)), 4, 32, 128, prompt)


if __name__ == '__main__':
    main()
