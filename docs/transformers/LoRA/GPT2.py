import torch
import torch.nn as nn
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# config from GPT
config = {
    "_name_or_path": "gpt2",
    "activation_function": "gelu_new",
    "architectures": [
        "GPT2LMHeadModel"
    ],
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 0,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_inner": None,
    "n_layer": 12,
    "n_positions": 1024,
    "reorder_and_upcast_attn": False,
    "resid_pdrop": 0.1,
    "scale_attn_by_inverse_layer_idx": False,
    "scale_attn_weights": True,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "task_specific_params": {
        "text-generation": {
            "do_sample": True,
            "max_length": 50
        }
    },
    "transformers_version": "4.42.4",
    "use_cache": True,
    "vocab_size": 50257
}


# from transformers
class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class HeadFFN(nn.Module):  # todo rename
    def __init__(self, dim):
        super().__init__()
        self.c_fc = Conv1D(dim, config['n_embd'])
        self.c_proj = Conv1D(config['n_embd'], dim)
        self.act = nn.functional.gelu

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = config['n_embd']
        self.num_heads = config['n_head']
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.c_att = Conv1D(config['n_embd'] * 3, config['n_embd'])
        self.c_proj = Conv1D(config['n_embd'], config['n_embd'])

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, hidden_states):
        batch_size, seq_length, _ = hidden_states.size()

        query, key, value = self.c_att(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,  # for the triangular mask
        )

        # todo why this?
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.embed_dim)

        attn_output = self.c_proj(attn_output)

        return attn_output


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = nn.LayerNorm(config['n_embd'], eps=config['layer_norm_epsilon'])
        self.attn = MultiHead()
        self.post_norm = nn.LayerNorm(config['n_embd'], eps=config['layer_norm_epsilon'])
        self.ffn = HeadFFN(config['n_embd'] * 4)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.pre_norm(hidden_states)

        attn_output = self.attn(hidden_states)

        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        feed_forward_output = self.ffn(hidden_states)
        hidden_states = feed_forward_output + residual

        return hidden_states


class GPTModel(nn.Module):
    # todo ignored token type embeds, past key values
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.position_embedding = nn.Embedding(config['n_positions'], config['n_embd'])

        self.blocks = nn.ModuleList([Block() for _ in range(config['n_layer'])])

        self.final_norm = nn.LayerNorm(config['n_embd'], eps=config['layer_norm_epsilon'])

        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

    def forward(self, input_ids):
        batch_size, input_shape = input_ids.size()

        token_embeddings = self.token_embedding(input_ids)  # B T C
        position_ids = torch.arange(input_shape)  # T C
        position_embeddings = self.position_embedding(position_ids)  # B T C

        hidden_states = token_embeddings + position_embeddings

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_norm(hidden_states)

        logits = self.lm_head(hidden_states)

        return logits


model = GPTModel()

state_dict = torch.load('transformed.pth')

missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
if missing_keys:
    print(f"Missing keys: {missing_keys}")
if unexpected_keys:
    print(f"Unexpected keys: {unexpected_keys}")

prompt = "hello how are you"
tokenized = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    model.eval()
    res = model(tokenized['input_ids'])

print(res)

output_ids = torch.argmax(res, dim=-1)

# Decode the token indices back to text
output_text = tokenizer.decode(output_ids[0])

# Print the tokens of the output
print(output_text)
