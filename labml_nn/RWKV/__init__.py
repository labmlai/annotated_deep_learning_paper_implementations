"""

---
title: Receptance Weighted Key Value (RWKV)
summary: >
  This implements the RWKV model 
  using PyTorch with explanations.
---

# Receptance Weighted Key Value (RWKV)

##TODO: make colab ?

This is a tutorial/implementation of RWKV
from paper [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/pdf/2305.13048.pdf)
in [PyTorch](https://pytorch.org/).

Full definition of a RWKV Language Model, all of it in this single file.
References:
1) the official RWKV PyTorch implementation released by Bo Peng:
https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py
"""


import math,time
import os
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


PREV_X_TIME = 0
NUM_STATE = 1
DEN_STATE = 2
MAX_STATE = 3
PREV_X_CHANNEL = 4

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# TODO: do I need this?
# learn from GPT-4
from unittest.mock import patch  
class CudaNotAvailable:  
    def __enter__(self):  
        self.patcher = patch("torch.cuda.is_available", return_value=False)  
        self.patcher.start()  
  
    def __exit__(self, exc_type, exc_value, traceback):  
        self.patcher.stop()  

#TODO: what is this? l2 norm?
# https://github.com/BlinkDL/RWKV-LM/blob/cca1b5e8e597cf40675882bb10b46287c844e35c/RWKV-v4/src/model.py#L21
class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss
    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

class ChannelMixing(nn.Module):
    """
    ## Channel Mixing
    """
    def __init__(self,config,layer_id):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.layer_id = layer_id
        
        n_embd = config.n_embd
        intermediate_size = (
            config.intermediate_size if config.intermediate_size is not None else 4 * n_embd
        )
        
        ## Learnable Matrix
        self.key_proj        = nn.Linear(n_embd,intermediate_size,bias=False)
        self.value_proj      = nn.Linear(intermediate_size,n_embd,bias=False)
        self.receptance_proj = nn.Linear(n_embd,n_embd,bias=False)
        
        ## Learnable Vector
        self.time_mix_key        = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, n_embd))

    def forward(self,x,state=None):
        # x = (Batch,Time,Channel)
        if state is not None:
            prev_x = state[self.layer_id,:,[PREV_X_CHANNEL],:]
            state[self.layer_id,:,[PREV_X_CHANNEL],:] = x
        else:
            prev_x = self.time_shift(x)
            
        """
        ### $r_t=W_r \cdot (\mu_r x_t + (1-\mu_r)x_{t-1})$
        """
        receptance = x * self.time_mix_receptance + prev_x * (1 - self.time_mix_receptance)
        receptance = self.receptance_proj(receptance)

        """
        ### $k_t=W_k \cdot (\mu_k x_t + (1-\mu_k)x_{t-1})$
        """
        key = x * self.time_mix_key + prev_x * (1 - self.time_mix_key)
        key = self.key_proj(key)

        """
        ### $V_t=W_v \cdot max(k_t,0)^2$
        """
        value = self.value_proj(torch.square(torch.relu(key)))

        """
        ### $o_t=\sigma(r_t) \odot v_t$
        """
        out = F.sigmoid(receptance) * value
        return out, state
    
"""
## Time Mixing
"""
class TimeMixing(nn.Module):
    def __init__(self,config,layer_id):
        super().__init__()
        self.config = config
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.layer_id = layer_id
        
        n_embd = config.n_embd
        attn_sz = n_embd

        ## learnable matrix
        self.key_proj        = nn.Linear(n_embd, attn_sz, bias=False)
        self.value_proj      = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance_proj = nn.Linear(n_embd, attn_sz, bias=False)
        self.output_proj     = nn.Linear(attn_sz, n_embd, bias=False)

        ## learnable vector
        self.time_decay          = nn.Parameter(torch.empty(attn_sz))
        self.time_first          = nn.Parameter(torch.empty(attn_sz))
        self.time_mix_key        = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_value      = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, n_embd))
    
    def forward(self,x,state=None):
        # x = (Batch,Time,Channel)
        if state is not None:
            prev_x = state[self.layer_id,:,[PREV_X_TIME],:]
            state[self.layer_id,:,[PREV_X_TIME],:] = x
        else:
            prev_x = self.time_shift(x)
        
        """
        ### $r_t=W_r \cdot (\mu_r x_t + (1-\mu_r)x_{t-1})$
        """
        receptance = x * self.time_mix_receptance + prev_x * (1 - self.time_mix_receptance)
        receptance = self.receptance_proj(receptance)

        """
        ### $k_t=W_k \cdot (\mu_k x_t + (1-\mu_k)x_{t-1})$
        """
        key = x * self.time_mix_key + prev_x * (1 - self.time_mix_key)
        key = self.key_proj(key)

        """
        ### $v_t=W_v \cdot (\mu_v x_t + (1-\mu_v)x_{t-1})$
        """
        value = x * self.time_mix_value + prev_x * (1 - self.time_mix_value)
        value = self.value_proj(value)

        """
        ### $wkv_t=\frac{\sum^{t-1}_{i=1}d^{-(t-1-i)w+k_i}v_i+e^{u+k_t}v_t}{\sum^{t-1}_{i=1}e^{-(t-1-i)w+k_i}+e^{u+k_t}}$
        """
        wkv, state  = self.wkv_function(key,value,use_customized_cuda_kernel=self.config.use_customized_cuda_kernel,state=state)
        
        """
        ### $o_t=W_o \cdot (\sigma(r_t) \odot wkv_t)$
        """
        rwkv = F.sigmoid(receptance) * wkv
        rwkv = self.output_proj(rwkv)
        
        return rwkv, state

    """
    ### helper function to forward(), does wkv calculation described in forward()
    uses custom cuda kernel. Actual implementation in pytorch shown in 
    """
    def wkv_function(self,key,value,use_customized_cuda_kernel,state=None):

        ## essentially, this customized cuda kernel delivers a faster for loop across time steps
        ## only for training and evaluating loss and ppl
        if state is None and use_customized_cuda_kernel:
            B, T, C = key.size()
            return WKVKernel.apply(B, T, C, self.time_decay, self.time_first, key, value), None
        
        ## raw wkv function (from Huggingface Implementation)
        ## only for generation
        else:    
            _, seq_length, _ = key.size()
            output = torch.zeros_like(key)

            debug_mode = False
            if state is None:
                ## only for debug purpose when use_customized_cuda_kernel=False and state is None
                debug_mode = True
                num_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
                den_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
                max_state = torch.zeros_like(key[:, 0], dtype=torch.float32) - 1e38
            else:
                num_state  = state[self.layer_id,:,NUM_STATE,:]
                den_state  = state[self.layer_id,:,DEN_STATE,:]
                max_state  = state[self.layer_id,:,MAX_STATE,:]

            time_decay = -torch.exp(self.time_decay)

            for current_index in range(seq_length):
                current_key = key[:, current_index].float()
                current_value = value[:, current_index]

                # wkv computation at time t
                max_for_output = torch.maximum(max_state, current_key + self.time_first)
                e1 = torch.exp(max_state - max_for_output)
                e2 = torch.exp(current_key + self.time_first - max_for_output)
                numerator = e1 * num_state + e2 * current_value
                denominator = e1 * den_state + e2
                output[:, current_index] = (numerator / denominator).to(output.dtype)

                # Update state for next iteration
                max_for_state = torch.maximum(max_state + time_decay, current_key)
                e1 = torch.exp(max_state + time_decay - max_for_state)
                e2 = torch.exp(current_key - max_for_state)
                num_state = e1 * num_state + e2 * current_value
                den_state = e1 * den_state + e2
                max_state = max_for_state
            
            if debug_mode:
                return output, None

            else:
                state[self.layer_id,:,NUM_STATE,:] = num_state
                state[self.layer_id,:,DEN_STATE,:] = den_state
                state[self.layer_id,:,MAX_STATE,:] = max_state

                return output, state

class Block(nn.Module):

    def __init__(self, config,layer_id):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = TimeMixing(config,layer_id)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ffn = ChannelMixing(config,layer_id)

    def forward(self, x, state = None):
        # state: [batch_size, 5 , n_embd]
        
        # time mixing
        residual = x
        x,state = self.attn(self.ln_1(x),state=state)
        x = x + residual
        
        # channel mixing
        residual = x
        x, state = self.ffn(self.ln_2(x),state=state)
        x = x + residual

        return x, state

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

class RWKV(nn.Module):

    def __init__(self, config,lr_init=0.0008):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.lr_init = lr_init ## used to initialize embedding parameters
        self.rwkv = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            ln_p = LayerNorm(config.n_embd, bias=config.bias),
            h = nn.ModuleList([Block(config,layer_id) for layer_id in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        if self.config.use_customized_cuda_kernel:
            ## load customized cuda kernel
            self.load_cuda_kernel(config.dtype)

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
                
    def forward(self, idx, targets=None, state=None, return_state=False):
        
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        x = self.rwkv.wte(idx)
        x = self.rwkv.ln_p(x)
        # x = self.rwkv.drop(x)
        for block_idx,block in enumerate(self.rwkv.h):
            x, state = block(x,state)
            if state is not None: ## in generation mode
                if (
                    self.config.rescale_every > 0 
                    and (block_idx + 1) % self.config.rescale_every == 0
                ):
                    x = x/2
        x = self.rwkv.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = L2Wrap.apply(loss,logits) # from RWKV-LM
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        
        if return_state:
            return logits, loss, state
        else:
            return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

    @classmethod
    def from_pretrained(cls, model_type,use_customized_cuda_kernel=True,dtype="float16"):
        assert model_type in {
            'RWKV/rwkv-4-169m-pile',
            "RWKV/rwkv-4-430m-pile",
            "RWKV/rwkv-4-1b5-pile",
            "RWKV/rwkv-4-3b-pile",
            "RWKV/rwkv-4-7b-pile",
            "RWKV/rwkv-raven-7b",
            "RWKV/rwkv-raven-1b5",
            "RWKV/rwkv-raven-3b",
            "RWKV/rwkv-4-14b-pile",
            }
        print("loading weights from pretrained RWKV: %s" % model_type)
                
        # init a huggingface/transformers model
        from transformers import RwkvForCausalLM,RwkvConfig
        hf_config = RwkvConfig.from_pretrained(model_type)
        with CudaNotAvailable(): ## avoid HF load kernel
            hf_model = RwkvForCausalLM.from_pretrained(model_type)

        # create a from-scratch initialized RWKV model
        config = {
            "vocab_size":50277,
            "n_layer":hf_config.num_hidden_layers,
            "n_embd":hf_config.hidden_size,
            "intermediate_size":hf_config.intermediate_size,
            "use_customized_cuda_kernel":use_customized_cuda_kernel,
            "dtype": dtype,
        }
        config = RWKVConfig(**config)
        model = RWKV(config)
        num_layers = config.n_layer
        ## create mapping from the parameter name in RWKV to that of HF-RWKV
        mapping = {
            "rwkv.wte.weight":"rwkv.embeddings.weight",
            "rwkv.ln_p.weight":"rwkv.blocks.0.pre_ln.weight",
            "rwkv.ln_p.bias":"rwkv.blocks.0.pre_ln.bias",
            "rwkv.ln_f.weight":"rwkv.ln_out.weight",
            "rwkv.ln_f.bias":"rwkv.ln_out.bias",
            "lm_head.weight":"head.weight",
            **{f"rwkv.h.{layer_id}.ln_{norm_id}.weight":f"rwkv.blocks.{layer_id}.ln{norm_id}.weight" for layer_id in range(num_layers) for norm_id in [1,2]},
            **{f"rwkv.h.{layer_id}.ln_{norm_id}.bias":f"rwkv.blocks.{layer_id}.ln{norm_id}.bias" for layer_id in range(num_layers) for norm_id in [1,2]},
            **{f"rwkv.h.{layer_id}.attn.{_type}":f"rwkv.blocks.{layer_id}.attention.{_type}" for layer_id in range(num_layers) for _type in ["time_decay","time_first",'time_mix_key','time_mix_value',"time_mix_receptance"]},
            **{f"rwkv.h.{layer_id}.attn.{_type}_proj.weight":f"rwkv.blocks.{layer_id}.attention.{_type}.weight" for layer_id in range(num_layers) for _type in ["key","value",'receptance',"output"]},
            **{f"rwkv.h.{layer_id}.ffn.{_type}":f"rwkv.blocks.{layer_id}.feed_forward.{_type}" for layer_id in range(num_layers) for _type in ['time_mix_key',"time_mix_receptance"]},
            **{f"rwkv.h.{layer_id}.ffn.{_type}_proj.weight":f"rwkv.blocks.{layer_id}.feed_forward.{_type}.weight" for layer_id in range(num_layers) for _type in ["key","value",'receptance']},
        }

        mapped_set = [mapping[x] for x in model.state_dict().keys()]
        assert set(mapped_set) == set(hf_model.state_dict().keys())
        sd = model.state_dict()
        hf_sd = hf_model.state_dict()

        for k1,k2 in mapping.items():
            assert sd[k1].shape == hf_sd[k2].shape,(k1,k2)
            sd[k1].copy_(hf_sd[k2])
        return model

    # def configure_optimizers(self,weight_decay,learning_rate,betas,device_type):
    #     # lr_1x = set()
    #     # lr_2x = set()
    #     # lr_3x = set()
    #     # for n, p in self.named_parameters():
    #     #     if "time_mix" in n:lr_1x.add(n)
    #     #     elif "time_decay" in n:lr_2x.add(n)
    #     #     elif "time_first" in n:lr_3x.add(n)
    #     #     else:lr_1x.add(n)
    #     # lr_1x = sorted(list(lr_1x))
    #     # lr_2x = sorted(list(lr_2x))
    #     # lr_3x = sorted(list(lr_3x))
        
    #     # param_dict = {n: p for n, p in self.named_parameters()}
    #     # optim_groups = [
    #     #     {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
    #     #     {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
    #     #     {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
    #     # ]

    #     optim_groups = [{"params": [p for n, p in self.named_parameters()], "weight_decay": 0.0},]
    #     fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    #     use_fused = fused_available and device_type == 'cuda'
    #     extra_args = dict(fused=True) if use_fused else dict()
    #     optimizer = torch.optim.Adam(optim_groups, lr=learning_rate, betas=betas, eps=1e-8, weight_decay=weight_decay,amsgrad=False,**extra_args)

    #     return optimizer
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

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see RWKV paper Appendix C as ref: https://arxiv.org/abs/2305.13048
        cfg = self.config
        L, V, D = cfg.n_layer, cfg.vocab_size, cfg.n_embd
        # Note there is a typo in the RWKV paper. Forward pass is 2*fn, forward
        # and backward is 6*fn.
        flops_per_token = 2*(V*D + 13*(V**2)*L)
        flops_per_fwdbwd = 3*flops_per_token
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf
        if cfg.dtype == 'bfloat16':
            flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        elif cfg.dtype == 'float16':
            flops_promised = 312e12 # A100 GPU float16 peak flops is 312 TFLOPS
        else: #dtype == float32
            flops_promised = 19.5e12 # A100 GPU float32 peak flops is 19.5 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def init_state(self,batch_size,device):

        n_state = len([PREV_X_TIME,NUM_STATE,DEN_STATE,MAX_STATE,PREV_X_CHANNEL])
        state = torch.zeros(
            (self.config.n_layer,batch_size,n_state,self.config.n_embd),
            dtype=torch.float32, device=device,
        )
        state[:,:,MAX_STATE,:] -= 1e30
        
        return state

    def scale_parameters(self):
        if self.config.rescale_every > 0:
            with torch.no_grad():
                for block_id,block in enumerate(self.rwkv.h):
                    block.attn.output_proj.weight.div_(2 ** int(block_id // self.config.rescale_every))
                    block.ffn.value_proj.weight.div_(2 ** int(block_id // self.config.rescale_every))
            self.scaled = True

    def unscale_parameters(self):
        if self.config.rescale_every > 0 and self.scaled:
            with torch.no_grad():
                for block_id,block in enumerate(self.rwkv.h):
                    block.attn.output_proj.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                    block.ffn.value_proj.weight.mul_(2 ** int(block_id // self.config.rescale_every))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        idx: (batch_size,seq_len)
        """
        batch_size,seq_len = idx.shape
        state = self.init_state(batch_size,idx.device)
        for seq_id in range(seq_len):
            logits, _, state = self(idx[:,[seq_id]], state = state, return_state=True)
        
        for _ in range(max_new_tokens):    
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            logits, _, state = self(idx_next, state=state, return_state=True)
        return idx

    def load_cuda_kernel(self,dtype):
        
        from torch.utils.cpp_extension import load
        T_MAX = self.config.block_size
        RWKV_FLOAT_MODE = dtype
        if RWKV_FLOAT_MODE == "bfloat16":
            wkv_cuda = load(name=f"wkv_{T_MAX}_bf16", sources=["cuda/wkv_op_bf16.cpp", "cuda/wkv_cuda_bf16.cu"], verbose=True, extra_cuda_cflags=["-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
            class WKV(torch.autograd.Function):
                @staticmethod
                def forward(ctx, B, T, C, w, u, k, v):
                    ctx.B = B
                    ctx.T = T
                    ctx.C = C
                    assert T <= T_MAX
                    assert B * C % min(C, 32) == 0
                    w = -torch.exp(w.float().contiguous())
                    u = u.contiguous().bfloat16()
                    k = k.contiguous()
                    v = v.contiguous()
                    y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
                    wkv_cuda.forward(B, T, C, w, u, k, v, y)
                    ctx.save_for_backward(w, u, k, v, y)
                    return y
                @staticmethod
                def backward(ctx, gy):
                    B = ctx.B
                    T = ctx.T
                    C = ctx.C
                    assert T <= T_MAX
                    assert B * C % min(C, 32) == 0
                    w, u, k, v, y = ctx.saved_tensors
                    gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
                    gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
                    gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
                    gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
                    wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
                    gw = torch.sum(gw, dim=0)
                    gu = torch.sum(gu, dim=0)
                    return (None, None, None, gw, gu, gk, gv)
        else:
            wkv_cuda = load(name=f"wkv_{T_MAX}", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"], verbose=True, extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
            class WKV(torch.autograd.Function):
                @staticmethod
                def forward(ctx, B, T, C, w, u, k, v):
                    ctx.B = B
                    ctx.T = T
                    ctx.C = C
                    assert T <= T_MAX
                    assert B * C % min(C, 32) == 0
                    if "32" in RWKV_FLOAT_MODE:
                        w = -torch.exp(w.contiguous())
                        u = u.contiguous()
                        k = k.contiguous()
                        v = v.contiguous()
                    else:
                        w = -torch.exp(w.float().contiguous())
                        u = u.float().contiguous()
                        k = k.float().contiguous()
                        v = v.float().contiguous()
                    y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
                    wkv_cuda.forward(B, T, C, w, u, k, v, y)
                    ctx.save_for_backward(w, u, k, v, y)
                    if "32" in RWKV_FLOAT_MODE:
                        return y
                    elif RWKV_FLOAT_MODE == "float16":
                        return y.half()
                
                @staticmethod
                def backward(ctx, gy):
                    B = ctx.B
                    T = ctx.T
                    C = ctx.C
                    assert T <= T_MAX
                    assert B * C % min(C, 32) == 0
                    w, u, k, v, y = ctx.saved_tensors
                    gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
                    gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
                    gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
                    gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
                    if "32" in RWKV_FLOAT_MODE:
                        wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
                    else:
                        wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.float().contiguous(), gw, gu, gk, gv)
                    gw = torch.sum(gw, dim=0)
                    gu = torch.sum(gu, dim=0)
                    if "32" in RWKV_FLOAT_MODE:
                        return (None, None, None, gw, gu, gk, gv)
                    elif RWKV_FLOAT_MODE == "float16":
                        return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())

        global WKVKernel
        WKVKernel = WKV 