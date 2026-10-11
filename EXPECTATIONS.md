# Expectations – NLP Course Project HS26

## Project 1: Word Embeddings and RNNs

Two model architectures will be evaluated:

1. Pretrained word embeddings with a two-layer classifier
   using ReLU activation.
2. The same pretrained word embeddings with a two-layer
   LSTM or GRU and a two-layer classifier using ReLU.

In both architectures, the pretrained word embeddings
remain frozen during training.

### Expected Test Accuracy

| Model | Expected Best Test Accuracy |
|-------|-----------------------------|
| Random baseline | 50% |
| Word embeddings + classifier | 55–60% |
| Word embeddings + RNN | 70-75% |

These values represent predictions made before conducting
the experiments, not measured results.

### Reasoning

#### Word Embeddings + Classifier

I expect the word embedding model to perform slightly better than random guessing, since pretrained embeddings capture textual similarity reasonably well.

However, reasoning requires more than similarity alone, so this approach will have its limitations.

---

#### Word Embeddings + RNN

I expect the RNN-based model to outperform the classifier
that uses word embeddings alone.

Unlike aggregated word embeddings, an LSTM or GRU can process
sequences of tokens and incorporate word order and contextual information.

This should help the model distinguish between hypotheses
that contain similar words but describe different events.

Never the less since we saw in the lectures that text created by RNN are mostly gibberish that looks like english.
I would still say that the RNN will not perform too good.
