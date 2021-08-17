# [Masked Language Model (MLM)](https://nn.labml.ai/transformers/mlm/index.html)

This is a [PyTorch](https://pytorch.org) implementation of Masked Language Model (MLM)
 used to pre-train the BERT model introduced in the paper
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://papers.labml.ai/paper/1810.04805).

## BERT Pretraining

BERT model is a transformer model.
The paper pre-trains the model using MLM and with next sentence prediction.
We have only implemented MLM here.

### Next sentence prediction

In *next sentence prediction*, the model is given two sentences `A` and `B` and the model
makes a binary prediction whether `B` is the sentence that follows `A` in the actual text.
The model is fed with actual sentence pairs 50% of the time and random pairs 50% of the time.
This classification is done while applying MLM. *We haven't implemented this here.*

## Masked LM

This masks a percentage of tokens at random and trains the model to predict
the masked tokens.
They **mask 15% of the tokens** by replacing them with a special `[MASK]` token.

The loss is computed on predicting the masked tokens only.
This causes a problem during fine-tuning and actual usage since there are no `[MASK]` tokens
 at that time.
Therefore we might not get any meaningful representations.

To overcome this **10% of the masked tokens are replaced with the original token**,
and another **10% of the masked tokens are replaced with a random token**.
This trains the model to give representations about the actual token whether or not the
input token at that position is a `[MASK]`.
And replacing with a random token causes it to
give a representation that has information from the context as well;
because it has to use the context to fix randomly replaced tokens.

## Training

MLMs are harder to train than autoregressive models because they have a smaller training signal.
i.e. only a small percentage of predictions are trained per sample.

Another problem is since the model is bidirectional, any token can see any other token.
This makes the "credit assignment" harder.
Let's say you have the character level model trying to predict `home *s where i want to be`.
At least during the early stages of the training, it'll be super hard to figure out why the
replacement for `*` should be `i`, it could be anything from the whole sentence.
Whilst, in an autoregressive setting the model will only have to use `h` to predict `o` and
`hom` to predict `e` and so on. So the model will initially start predicting with a shorter context first
and then learn to use longer contexts later.
Since MLMs have this problem it's a lot faster to train if you start with a smaller sequence length
initially and then use a longer sequence length later.

Here is [the training code](https://nn.labml.ai/transformers/mlm/experiment.html) for a simple MLM model.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/3a6d22b6c67111ebb03d6764d13a38d1)
