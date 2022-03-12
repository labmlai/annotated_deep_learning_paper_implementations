"""
---
title: Retrieval-Enhanced Transformer (Retro)
summary: >
  This is a PyTorch implementation/tutorial of the paper
  Improving language models by retrieving from trillions of tokens.
  It builds a key-value database of chunks of text and retrieves and uses them when
  making predictions.
---

# Retrieval-Enhanced Transformer (Retro)

This is a [PyTorch](https://pytorch.org) implementation of the paper
[Improving language models by retrieving from trillions of tokens](https://papers.labml.ai/paper/2112.04426).

It builds a database of chunks of text.
It is a key-value database where the keys are indexed by the BERT embeddings of the chunks.
They use a frozen pre-trained BERT model to calculate these embeddings.
The values are the corresponding chunks and an equal length of text proceeding that chunk.

Then the model retrieves text similar (nearest neighbors) to the input to the model from this database.
These retrieved texts are used to predict the output.

Since we use a frozen BERT model for retrieval we can pre-calculate all the nearest neighbors for the
training dataset.
This speeds up the training process.

Components:

* [BERT embeddings](bert_embeddings.html): Code to get BERT embeddings of chunks of text.
* [Key-value database](database.html): Build and retrieve chunks
* [Model](model.html)
* [Dataset](dataset.html): Pre-calculate the nearest neighbors
* [Training code](train.html)

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/3113dd3ea1e711ec85ee295d18534021)
"""