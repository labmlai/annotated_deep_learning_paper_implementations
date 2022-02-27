from typing import List

import torch
from transformers import BertTokenizer, BertModel

from labml import lab, monit


class BERTChunkEmbeddings:
    def __init__(self, device: torch.device):
        with monit.section('Load BERT tokenizer'):
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                           cache_dir=str(
                                                               lab.get_data_path() / 'cache' / 'bert-tokenizer'))

        with monit.section('Load BERT model'):
            self.model = BertModel.from_pretrained("bert-base-uncased",
                                                   cache_dir=str(lab.get_data_path() / 'cache' / 'bert-model'))

            self.model.to(device)

        self.device = device

    def __call__(self, chunks: List[str]):
        tokens = self.tokenizer(chunks, return_tensors='pt', add_special_tokens=False, padding=True)

        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=tokens['token_type_ids'].to(self.device))

        state = output['last_hidden_state']
        emb = (state * attention_mask[:, :, None]).sum(dim=1) / attention_mask[:, :, None].sum(dim=1)

        return emb


def _test():
    from labml.logger import inspect

    device = torch.device('cuda:0')
    bert = BERTChunkEmbeddings(device)

    text = ["Replace me by any text you'd like. abracadabra Jayasiri",
            "Second sentence"]

    encoded_input = bert.tokenizer(text, return_tensors='pt', add_special_tokens=False, padding=True)

    inspect(encoded_input, _expand=True)

    output = bert.model(input_ids=encoded_input['input_ids'].to(device),
                        attention_mask=encoded_input['attention_mask'].to(device),
                        token_type_ids=encoded_input['token_type_ids'].to(device))

    inspect({'last_hidden_state': output['last_hidden_state'],
             'pooler_output': output['pooler_output']},
            _expand=True)

    inspect(bert.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0]), _n=-1)
    inspect(bert.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][1]), _n=-1)

    bert(text)


if __name__ == '__main__':
    _test()
