import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

state_dict = model.state_dict()

mapping = {
    'transformer.wte.weight': 'token_embedding.weight',
    'transformer.wpe.weight': 'position_embedding.weight',
    'transformer.ln_f.weight': 'final_norm.weight',
    'transformer.ln_f.bias': 'final_norm.bias',
    'lm_head.weight': 'lm_head.weight'
}

for i in range(12):
    mapping[f'transformer.h.{i}.ln_1.weight'] = f'blocks.{i}.pre_norm.weight'
    mapping[f'transformer.h.{i}.ln_1.bias'] = f'blocks.{i}.pre_norm.bias'
    mapping[f'transformer.h.{i}.attn.c_attn.weight'] = f'blocks.{i}.attn.c_att.weight'
    mapping[f'transformer.h.{i}.attn.c_attn.bias'] = f'blocks.{i}.attn.c_att.bias'
    mapping[f'transformer.h.{i}.attn.c_proj.weight'] = f'blocks.{i}.attn.c_proj.weight'
    mapping[f'transformer.h.{i}.attn.c_proj.bias'] = f'blocks.{i}.attn.c_proj.bias'
    mapping[f'transformer.h.{i}.ln_2.weight'] = f'blocks.{i}.post_norm.weight'
    mapping[f'transformer.h.{i}.ln_2.bias'] = f'blocks.{i}.post_norm.bias'
    mapping[f'transformer.h.{i}.mlp.c_fc.weight'] = f'blocks.{i}.ffn.c_fc.weight'
    mapping[f'transformer.h.{i}.mlp.c_fc.bias'] = f'blocks.{i}.ffn.c_fc.bias'
    mapping[f'transformer.h.{i}.mlp.c_proj.weight'] = f'blocks.{i}.ffn.c_proj.weight'
    mapping[f'transformer.h.{i}.mlp.c_proj.bias'] = f'blocks.{i}.ffn.c_proj.bias'

new_state_dict = {}
for old_key, new_key in mapping.items():
    if old_key in state_dict:
        new_state_dict[new_key] = state_dict[old_key]

torch.save(new_state_dict, 'transformed.pth')
