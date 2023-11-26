"""
---
title: CLIP Text Embedder
summary: >
 CLIP embedder to get prompt embeddings for stable diffusion
---

# CLIP Text Embedder

This is used to get prompt embeddings for [stable diffusion](../index.html).
It uses HuggingFace Transformers CLIP model.
"""

from typing import List
import kornia
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel
from clip import load as load_clip
import torch

class CLIPTextEmbedder(nn.Module):
    """
    ## CLIP Text Embedder
    """

    # def __init__(self, version: str = "/other_models/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff", device="cuda:0", max_length: int = 77):
    
    def __init__(self, version: str = "openai/clip-vit-large-patch14", device="cuda:0", max_length: int = 77):

        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()
        # Load the tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        # Load the CLIP transformer
        self.transformer = CLIPTextModel.from_pretrained(version).eval()

        self.device = device
        self.max_length = max_length

    def forward(self, prompts: List[str]):
        """
        :param prompts: are the list of prompts to embed
        """
        # Tokenize the prompts
        batch_encoding = self.tokenizer(prompts, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        # Get token ids
        tokens = batch_encoding["input_ids"].to(self.device)
        # Get CLIP embeddings
        return self.transformer(input_ids=tokens).last_hidden_state


class CLIPImageEmbedder(nn.Module):
    """
    ## CLIP Image Embedder
    """
    # def __init__(self, model="/other_models/clip/ViT-B-32.pt", device='cuda:0'):
    def __init__(self, model="ViT-B/32", device='cuda:0'):
        super().__init__()
        
        self.model, _ = load_clip(name=model, device=device)
        self.model.eval()
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic', align_corners=True)
        x = (x + 1.) / 2.
        # re-normalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        out = self.model.encode_image(self.preprocess(x))
        # out = out.to(x.dtype)
        return out
    
