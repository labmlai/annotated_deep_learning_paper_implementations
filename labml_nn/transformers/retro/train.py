from labml import monit
from labml_nn.transformers.retro import model as retro


class Sampler:
    def __init__(self, tokenizer, model):
        pass

    def sample(self, prompt: str, sample_len):
        neighbors = []
        for i in range(sample_len):
            pass
            # retrieve neighbors if there are new chunks
            # append to neighbors

            # evaluate model
            # get the next token


class Trainer:
    epochs: int = 200
    batch_size: int = 1

    learning_rate = 0.0002
    adam_betas = (0.5, 0.999)
    decay_start = 100

    def __init__(self, model: retro.RetroModel):
        self.model = model
        # Load dataset
        # Initialize tokenizer
        # Load dataloaders

    def train(self):
        # Loop through epochs
        for epoch in monit.loop(self.epochs):
            pass
            # Loop through the train dataset
            # Tokenize strings
            # Train the model

            # Loop through the eval dataset
            # Tokenize strings
