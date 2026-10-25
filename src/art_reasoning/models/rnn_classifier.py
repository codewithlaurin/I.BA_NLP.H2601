"""Connect the recurrent ART encoder to the existing two-layer classifier.

The architecture, initialization, and output decisions are in docs/RNN.md.
"""

import torch
from torch import nn

from ..batching import TextBatch
from .classifier import Classifier
from .sequence_encoder import SequenceEncoder


class RNNClassifier(nn.Module):
    """Map a complete TextBatch to two hypothesis logits through one model.

    The supplied encoder determines the head's input width. Its embedding
    remains frozen; the recurrent layers and freshly initialized classifier
    remain trainable. Calling train(), eval(), or to() applies to both parts.
    """

    def __init__(
        self, encoder: SequenceEncoder, hidden_dim: int = 256, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = Classifier(
            input_dim=encoder.output_dim, hidden_dim=hidden_dim, dropout=dropout
        ).to(
            device=encoder.embedding.weight.device,
            dtype=encoder.embedding.weight.dtype,
        )

    def forward(self, batch: TextBatch) -> torch.Tensor:
        """Return raw logits shaped (B, 2), ready for cross-entropy loss."""
        return self.classifier(self.encoder(batch))

    @torch.no_grad()
    def predict_proba(self, batch: TextBatch) -> torch.Tensor:
        """Return softmax probabilities; call model.eval() to disable dropout."""
        return self.classifier.predict_proba(self.encoder(batch))
