import inspect
import math

import torch
import torch.nn as nn
from labml_nn.RWKV.configs import RWKVConfigs

from labml_nn.RWKV import RWKV
from labml_nn.RWKV import TimeMixing
from labml import experiment
from labml.configs import option
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    This inherits from
    [`NLPAutoRegressionConfigs`](../../experiments/nlp_autoregression.html#NLPAutoRegressionConfigs)
    """

    # RWKV model
    model: RWKV

    rwkv: RWKVConfigs
    # number of warmup iterations
    warmup_iters: int = 2000
    # total number of training iterations
    max_iters: int = 600000
    # weight decay
    weight_decay: float = 1e-1
    # Custom optimizer
    beta1: float = 0.9
    beta2: float = 0.95
    optimizer = 'rwkv_optimizer'


@option(Configs.rwkv, 'RWKV')
def _rwkv_configs(c: Configs):
    """
    ### RWKV configurations
    """

    # We use our
    # [configurable RWKV implementation](../configs.html#RWKVConfigs)
    conf = RWKVConfigs()
    # Set the vocabulary sizes for embeddings and generating logits
    conf.n_src_vocab = c.n_tokens
    conf.n_tgt_vocab = c.n_tokens

    return conf


def _init_weights(module, rwkv: RWKVConfigs):
    # initialize Vector Parameters in TimeMixing
    if isinstance(module, TimeMixing):
        layer_id = module.layer_id
        n_layer = module.n_layer
        n_embd = module.n_embd
        attn_sz = n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            decay_speed = torch.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            module.time_decay = nn.Parameter(decay_speed)

            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5
            module.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)
            module.time_mix_key = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            module.time_mix_value = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            module.time_mix_receptance = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))


@option(Configs.model)
def _model(c: Configs):
    """
    Create RWKV model and initialize weights
    """
    m = RWKV(c.rwkv).to(c.device)

    # Apply custom weight initialization
    m.apply(_init_weights, c.rwkv)

    return m


@option(NLPAutoRegressionConfigs.optimizer)
def _configure_optimizers(c: NLPAutoRegressionConfigs):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in c.model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': c.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and c.device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=c.learning_rate, betas=c.betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer


def main():
    # Create experiment
    experiment.create(name="RWKV")
    # Create configs
    conf = Configs()
    print(conf.model)
    # Override configurations
    experiment.configs(conf, {
        # Use character level tokenizer
        'tokenizer': 'character',
        # Prompt separator is blank
        'prompt_separator': '',
        # Starting prompt for sampling
        'prompt': 'It is ',
        # Use Tiny Shakespeare dataset
        'text': 'tiny_shakespeare',

        # Use a context size of $128$
        'seq_len': 128,
        # Train for $32$ epochs
        'epochs': 32,
        # Batch size $128$
        'batch_size': 128,
        # Switch between training and validation for $10$ times
        # per epoch
        'inner_iterations': 10,

        'rwkv.block_size': 1024,
        # model
        'rwkv.n_layer': 12,
        'rwkv.n_heads': 12,
        'rwkv.n_embd': 768
    })

    print(conf.model)
    # Set models for saving and loading
    experiment.add_pytorch_models({'model': conf.model})

    # Start the experiment
    with experiment.start():
        # Run training
        conf.run()


#
if __name__ == '__main__':
    main()
