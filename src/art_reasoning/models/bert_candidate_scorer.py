"""Score both ART hypotheses with a shared BERT encoder and linear head."""

import torch
from torch import nn
from transformers import AutoTokenizer, BertConfig, BertModel, PreTrainedTokenizerBase

from ..transformer_data import CandidateBatch


class BertCandidateScorer(nn.Module):
    def __init__(self, encoder: BertModel) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(encoder.config.hidden_dropout_prob)
        # A shared bias would cancel when comparing the two candidate logits.
        self.scorer = nn.Linear(encoder.config.hidden_size, 1, bias=False)
        nn.init.normal_(self.scorer.weight, std=encoder.config.initializer_range)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str = "google-bert/bert-base-cased",
        *,
        pretrained: bool = True,
        revision: str | None = None,
        dropout: float | None = None,
    ) -> "BertCandidateScorer":
        """Keep the architecture fixed; only the encoder initialization differs.

        dropout replaces the checkpoint's dropout probability (0.1 for BERT) in the
        embeddings, attention, and hidden layers; the scoring head uses the same value.
        """
        overrides = {} if dropout is None else {
            "hidden_dropout_prob": dropout, "attention_probs_dropout_prob": dropout,
        }
        config = BertConfig.from_pretrained(checkpoint, revision=revision, **overrides)
        if pretrained:
            encoder = BertModel.from_pretrained(
                checkpoint, config=config, revision=revision, add_pooling_layer=False
            )
        else:
            encoder = BertModel(config, add_pooling_layer=False)
        return cls(encoder)

    def forward(self, batch: CandidateBatch) -> torch.Tensor:
        """Return raw (B, 2) logits for cross_entropy(logits, batch.labels)."""
        batch_size, choices, length = batch.inputs["input_ids"].shape
        if choices != 2:
            raise ValueError("Each ART example must have exactly two candidates")
        if length > self.encoder.config.max_position_embeddings:
            raise ValueError("Candidate length exceeds BERT's context limit")
        # Flatten candidates into one encoder batch, without attention between them.
        inputs = {
            name: tensor.reshape(batch_size * choices, length)
            for name, tensor in batch.inputs.items()
        }
        output = self.encoder(**inputs)
        cls_states = output.last_hidden_state[:, 0]
        return self.scorer(self.dropout(cls_states)).reshape(batch_size, choices)


def load_bert_components(
    *,
    pretrained: bool,
    checkpoint: str = "google-bert/bert-base-cased",
    revision: str | None = "cd5ef92a9fb2f889e972770a36d4ed042daf221e",
    dropout: float | None = None,
) -> tuple[BertCandidateScorer, PreTrainedTokenizerBase]:
    """Load either initialization with the same checkpoint's fixed tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    model = BertCandidateScorer.from_checkpoint(
        checkpoint, pretrained=pretrained, revision=revision, dropout=dropout
    )
    return model, tokenizer
