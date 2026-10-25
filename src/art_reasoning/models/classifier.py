"""Two-layer classifier for fixed-size mean-pooled or recurrent ART features."""

import torch
from torch import nn

from .classifier_input import ClassifierInput


class Classifier(nn.Module):
    """Map (B, input_dim) features to (B, 2) unnormalized hypothesis scores.

    Architecture: Linear -> ReLU -> Dropout -> Linear (two trainable layers).
    forward() returns raw logits for CrossEntropyLoss. predict_proba() applies
    softmax over the two hypotheses for probability output.
    """

    def __init__(
        self, input_dim: int = ClassifierInput.output_dim, hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 (inclusive) and 1")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.output_activation = nn.Softmax(dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits for training with cross-entropy loss."""
        return self.layers(features)

    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Return (B, 2) probabilities; call eval() first to disable dropout."""
        return self.output_activation(self(features))
