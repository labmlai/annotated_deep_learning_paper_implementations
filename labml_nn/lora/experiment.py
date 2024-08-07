import torch
from labml import lab, monit, tracker
from labml.configs import BaseConfigs, option
from labml.utils.download import download_file
from labml_helpers.device import DeviceConfigs
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from labml_nn.lora.gpt2 import GPTModel


class Configs(BaseConfigs):
    device: torch.device = DeviceConfigs()
    layer_norm_epsilon: float = 1e-05
    n_embed: int = 768
    n_layer: int = 12
    n_positions: int = 1024
    vocab_size: int = 50257
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    context_len: int = 512
    r: int = 32

    text: TensorDataset = "tiny_shakespeare"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model: GPTModel
    optimizer: torch.optim.Adam
    criterion = torch.nn.CrossEntropyLoss()
    data_loader: DataLoader

    def _load_pretrained_weights(self):
        hf_model = AutoModelForCausalLM.from_pretrained("gpt2")

        state_dict = hf_model.state_dict()

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

        # transpose weight matrices of convo 1d layers to use linear layers instead
        convo_layers = ([f'blocks.{i}.ffn.c_fc.weight' for i in range(12)] +
                        [f'blocks.{i}.ffn.c_proj.weight' for i in range(12)] +
                        [f'blocks.{i}.attn.c_att.weight' for i in range(12)] +
                        [f'blocks.{i}.attn.c_proj.weight' for i in range(12)])

        for layer in convo_layers:
            new_state_dict[layer] = torch.transpose(new_state_dict[layer], 0, 1)

        self.model.load_state_dict(new_state_dict, strict=False)  # state dict does not have lora weights

        del hf_model
        del state_dict
        del new_state_dict

    def initialize(self):
        self.model = GPTModel(
            layer_norm_epsilon=self.layer_norm_epsilon,
            n_embd=self.n_embed,
            n_layer=self.n_layer,
            n_positions=self.n_positions,
            vocab_size=self.vocab_size,
            r=self.r,
            device=self.device
        ).to(self.device)
        self._load_pretrained_weights()

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        self.data_loader = DataLoader(self.text, batch_size=self.batch_size, shuffle=True)

    def run(self):
        for _ in monit.loop(self.epochs):
            for i, batch in monit.enum('Train', self.data_loader):
                inputs = batch[0]
                inputs = inputs.to(self.device)
                labels = inputs.clone()

                outputs = self.model(inputs)

                shift_logits = outputs[..., :-1, :]
                shift_labels = labels[..., 1:]

                loss = self.criterion(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                tracker.add({'loss': loss})

                tracker.save()
                tracker.add_global_step()
            tracker.new_line()


@option(Configs.text)
def tiny_shakespeare(c: Configs):
    """
    ### Tiny Shakespeare dataset

    It will download from the url if not present
    """
    path = lab.get_data_path() / 'tiny_shakespeare.txt'
    if not path.exists():
        download_file("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", path)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = c.tokenizer.encode(text)
    num_batches = len(tokens) // (c.batch_size * c.context_len)
    tokens = tokens[:num_batches * c.batch_size * c.context_len]
    input_ids = torch.tensor(tokens).view(-1, c.context_len)
    return TensorDataset(input_ids)
