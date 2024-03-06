import math,time
import os
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from labml import experiment
from labml.configs import option
from labml_nn.RWKV import RWKV
from labml_nn.RWKV import TimeMixing
from labml_nn.RWKV import ChannelMixing
from contextlib import nullcontext

out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# model
model_type = 'gpt'
use_customized_cuda_kernel = True

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
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

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

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = [torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix]
    y = [torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix]
    
    x = torch.stack(x)
    y = torch.stack(y)

    x, y = x.to(device), y.to(device)
    return x, y

def train_step(model):
    X,Y = get_batch('train')
    logits, loss = model(X,Y)
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
model_args = dict(n_layer=n_layer, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dtype=dtype,use_customized_cuda_kernel=use_customized_cuda_kernel)

model = RWKV_experiment(RWKVConfig(**model_args))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
args = RWKVConfig()
print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
model = RWKV_experiment(args)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
iter_num = 0


while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = get_batch('train')
    # backward pass, with gradient scaling if training in fp16
    scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
