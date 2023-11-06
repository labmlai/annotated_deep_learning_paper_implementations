import math,time
import os
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from labml import experiment
from labml.configs import option
from labml_nn.RWKV import RWKV
from labml_nn.RWKV import TimeMixing
from labml_nn.RWKV import ChannelMixing


@dataclass
class RWKVConfig:
    block_size: int = 1024 # same as nanoGPT
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_embd: int = 768
    bias: bool = True # bias in LayerNorms, in RWKV, all bias in Linear is False
    intermediate_size: int = None # intermediate_size in channel-mixing
    use_customized_cuda_kernel: bool = True
    dtype: str = "float16" ## bfloat16 is not supported in V100
    rescale_every: int = 6 ## mysterious trick, only applies when inference


class RWKV_experiment(RWKV):

    def __init__(self):
        super.__init__()
        self.apply(self._init_weights)
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the token embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.rwkv.wte.weight.numel()
        return n_params

    def _init_weights(self, module):

        ## initialize Vector Parameters in TimeMixing
        if isinstance(module,TimeMixing):
            layer_id = module.layer_id
            n_layer = self.config.n_layer
            n_embd = self.config.n_embd
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
        
        ## initialize Vector Parameters in ChannelMixing
        elif isinstance(module,ChannelMixing):
            layer_id = module.layer_id
            n_layer = self.config.n_layer
            n_embd = self.config.n_embd
            
            with torch.no_grad():  # fancy init of time_mix
                ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, n_embd)
                for i in range(n_embd):
                    ddd[0, 0, i] = i / n_embd
                module.time_mix_key = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                module.time_mix_receptance = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
        ## initialize Linear Layer and Embedding Layer
        elif isinstance(module,(nn.Embedding,nn.Linear)):
            weight = module.weight
            shape = weight.shape
            gain = 1.0
            scale = 1.0
            
            ## get the current name of the parameters
            for _name,_parameters in self.named_parameters():
                if id(_parameters) == id(weight):
                    current_module_name = _name
            
            # print(current_module_name)

            ## Embedding
            if isinstance(module, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                scale = -1 * self.lr_init

            ## Linear
            elif isinstance(module,nn.Linear):
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                
                ## initialize some matrix to be all ZEROS
                for name in [".attn.key_proj.", ".attn.receptance_proj.", ".attn.output_proj.", 
                             ".ffn.value_proj.", ".ffn.receptance_proj."]:
                    if name in current_module_name:
                        scale = 0
                
                if current_module_name == 'lm_head.weight':
                    scale = 0.5

            if scale == 0:
                nn.init.zeros_(weight)
            elif scale < 0:
                nn.init.uniform_(weight, a=scale, b=-scale)
            else:
                nn.init.orthogonal_(weight, gain=gain * scale)

print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
model = RWKV_experiment(args)

print(f'\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)')
init_state = None
for token in tokenizer.encode(context).ids:
    init_out, init_state = model.forward(token, init_state)

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
    all_tokens = []
    out_last = 0
    out, state = init_out.clone(), init_state.clone()
    for i in range(LENGTH_PER_TRIAL):
        token = sample_logits(out, TEMPERATURE, TOP_P)
        all_tokens += [token]
        tmp = tokenizer.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
            print(tmp, end="", flush=True)
            out_last = i + 1
        out, state = model.forward(token, state)       
print('\n')