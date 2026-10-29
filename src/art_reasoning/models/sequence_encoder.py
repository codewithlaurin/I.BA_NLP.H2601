"""Encode ART's four text fields with frozen FastText and a shared stacked RNN.

Architecture and padding decisions are documented in docs/RNN.md.
"""

from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ..batching import TextBatch
from ..data import TEXT_FIELDS
from ..embeddings import EMBEDDING_DIM


class SequenceEncoder(nn.Module):
    """Return top-layer final states shaped (B, 4 * hidden_size * directions).

    One two-layer GRU (default) or LSTM shares parameters
    across all fields. Each field starts with a fresh zero recurrent state.
    Recorded lengths exclude padding; empty fields return exact zero vectors.
    The embedding is frozen, while both recurrent layers remain trainable.
    ``dropout`` applies between recurrent layers, separately from head dropout.
    Direction defaults to forward-only; bidirectional states concatenate
    forward then backward for each field, before concatenating the four fields.
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        hidden_size: int = 128,
        rnn_type: Literal["gru", "lstm"] = "gru",
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if embedding.embedding_dim != EMBEDDING_DIM:
            raise ValueError(f"Expected {EMBEDDING_DIM}-dimensional embeddings")
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if rnn_type not in ("gru", "lstm"):
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 (inclusive) and 1")
        if not isinstance(bidirectional, bool):
            raise ValueError("bidirectional must be a boolean")

        self.embedding = embedding
        self.embedding.requires_grad_(False)
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.field_output_dim = hidden_size * self.num_directions
        self.output_dim = len(TEXT_FIELDS) * self.field_output_dim
        recurrent_class = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = recurrent_class(
            input_size=EMBEDDING_DIM,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        ).to(device=embedding.weight.device, dtype=embedding.weight.dtype)

    def forward(self, batch: TextBatch) -> torch.Tensor:
        """Encode original tokens in TEXT_FIELDS order, without using labels."""
        ids = batch.token_ids
        if ids.ndim != 3 or ids.shape[1] != len(TEXT_FIELDS):
            raise ValueError("Expected token IDs shaped (batch_size, 4, sequence_length)")
        if batch.lengths.shape != ids.shape[:2]:
            raise ValueError("Expected one original length per example and text field")
        if batch.lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("Original lengths must be integer tensors")

        # Packing requires CPU lengths, even when IDs and weights are on a GPU.
        lengths = batch.lengths.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        if ((lengths < 0) | (lengths > ids.shape[2])).any().item():
            raise ValueError("Original lengths must be between zero and the padded width")

        features = self.embedding.weight.new_zeros((lengths.numel(), self.field_output_dim))
        nonempty_cpu = lengths.nonzero(as_tuple=True)[0]
        if nonempty_cpu.numel():
            nonempty = nonempty_cpu.to(ids.device)
            sequences = ids.reshape(lengths.numel(), ids.shape[2]).index_select(0, nonempty)
            vectors = self.embedding(sequences)
            packed = pack_padded_sequence(
                vectors, lengths.index_select(0, nonempty_cpu),
                batch_first=True, enforce_sorted=False,
            )
            # With unsorted packing, PyTorch restores h_n to the input order.
            _, state = self.rnn(packed)
            hidden = state[0] if self.rnn_type == "lstm" else state
            # h_n orders layer then direction. For a bidirectional top layer,
            # [-2] is forward and [-1] is backward; both summarize the full text.
            top_hidden = (
                torch.cat((hidden[-2], hidden[-1]), dim=-1)
                if self.bidirectional else hidden[-1]
            )
            features = features.index_copy(0, nonempty, top_hidden)

        return features.reshape(ids.shape[0], self.output_dim)
