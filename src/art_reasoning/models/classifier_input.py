"""Build fixed-size classifier inputs from frozen FastText word embeddings."""

import torch
from torch import nn

from ..batching import TextBatch
from ..data import TEXT_FIELDS
from ..embeddings import EMBEDDING_DIM


class ClassifierInput(nn.Module):
    """Mean-pool each text, then concatenate in TEXT_FIELDS order.

    A batch of token IDs shaped (B, 4, L) becomes features shaped (B, 1200).
    Padding is excluded from both the sum and denominator. Empty texts
    produce zero vectors. Every original token contributes; none is truncated.
    """

    output_dim = len(TEXT_FIELDS) * EMBEDDING_DIM

    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        if embedding.embedding_dim != EMBEDDING_DIM:
            raise ValueError(f"Expected {EMBEDDING_DIM}-dimensional embeddings")
        self.embedding = embedding
        self.embedding.requires_grad_(False)

    def forward(self, batch: TextBatch) -> torch.Tensor:
        """Return one feature vector per example, keeping labels separate."""
        if batch.token_ids.ndim != 3 or batch.token_ids.shape[1] != len(TEXT_FIELDS):
            raise ValueError("Expected token IDs shaped (batch_size, 4, sequence_length)")
        if batch.lengths.shape != batch.token_ids.shape[:2]:
            raise ValueError("Expected one original length per example and text field")

        vectors = self.embedding(batch.token_ids)  # (B, 4, L, 300)
        mask = batch.mask.unsqueeze(-1)  # (B, 4, L, 1)
        sums = vectors.masked_fill(~mask, 0).sum(dim=2)
        counts = batch.lengths.clamp_min(1).unsqueeze(-1).to(vectors.dtype)
        means = sums / counts  # (B, 4, 300); empty texts remain zero.
        return means.flatten(start_dim=1)  # (B, 1200)
