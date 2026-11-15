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

## Project 2: Transformers

Recorded on 2026-11-15 during planning, before running Project 2 experiments. These predictions take the already observed Project 1 results into account.

### Expected Test Accuracy

| Model | Expected Best Test Accuracy |
|-------|----------------------------|
| Random baseline | 50% |
| Randomly initialized Transformer | 50–55% |
| Pretrained Transformer | 60–70% |
| Prompted 8B+ LLM | 70–80% |

### Reasoning

#### Randomly Initialized Transformer

I expect performance close to random guessing, with a possible small improvement from learning task-specific patterns. The 16,801 training examples provide limited data for learning both useful language representations and the task from scratch, especially with a BERT-base-sized model.

#### Pretrained Transformer

I expect pretraining to provide a clear advantage over random initialization because the model starts with contextual language representations. Fine-tuning should help it compare the hypotheses against both observations. However, the small improvement over chance on the Project 1 test set makes me cautious about expecting a very high test accuracy.

#### Prompted LLM

I expect a LLM to perform best because its prior language knowledge should help assess which hypothesis plausibly connects the observations. Clear instructions and a few training examples may improve its answers, but subtle differences between hypotheses and invalid output formats can still cause errors. I do not assume that requesting step-by-step reasoning improves accuracy; that requires a validation comparison. The expected range is uncertain and depends on the selected prompt and generation settings.
