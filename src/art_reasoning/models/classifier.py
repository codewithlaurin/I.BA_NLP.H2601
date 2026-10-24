"""Two-layer classifier for the pooled ART text representations."""

import torch
from torch import nn

from .classifier_input import ClassifierInput


class Classifier(nn.Module):
    """Map (B, input_dim) features to (B, 2) unnormalized hypothesis scores.

    Architecture: Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 2).
    forward() returns raw logits for CrossEntropyLoss. predict_proba() applies
    softmax over the two hypotheses for probability output.
    """

    def __init__(
        self, input_dim: int = ClassifierInput.output_dim, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.output_activation = nn.Softmax(dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits for training with cross-entropy loss."""
        return self.layers(features)

    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Return (B, 2) probabilities summing to one for each example."""
        return self.output_activation(self(features))
