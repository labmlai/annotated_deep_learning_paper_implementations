from torch import nn


class Encoder(nn.Module):
    def __init__(self, n_vocab, d_model):
        super().__init__()

    def forward(self, h: torch.Tensor, ):

class Model(nn.Module):
    def __init__(self, n_vocab, d_model):
        super().__init__()

        self.emb = nn.Embedding(n_vocab, d_model)
        self.encoder = Encoder()