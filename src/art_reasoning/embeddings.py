"""Load frozen, 300-dimensional FastText embeddings for a project vocabulary."""

from pathlib import Path
from collections.abc import Iterable

import fasttext
import torch
from torch import nn

from .vocabulary import Vocabulary


EMBEDDING_DIM = 300


def load_fasttext_embeddings(
    vocabulary: Vocabulary, model_path: str | Path
) -> nn.Embedding:
    """Load a pretrained FastText .bin model into a frozen embedding layer.

    Each matrix row corresponds to the same ID in ``vocabulary``. Ordinary
    words use FastText's word vectors, including subword vectors for words
    outside its dictionary. <PAD> is zero; <UNK> is the mean of the ordinary
    vocabulary vectors, or zero when the vocabulary has no ordinary words.

    The returned layer owns only the vocabulary's embedding matrix, allowing
    the full FastText model to be released after construction.
    """
    model_path = Path(model_path).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"FastText model file not found: {model_path}")

    model = fasttext.load_model(str(model_path))
    dimension = model.get_dimension()
    if dimension != EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EMBEDDING_DIM}-dimensional FastText vectors, got {dimension}"
        )

    matrix = torch.zeros((len(vocabulary), EMBEDDING_DIM), dtype=torch.float32)
    for word, token_id in vocabulary.token_to_id.items():
        if token_id in (vocabulary.PAD_ID, vocabulary.UNK_ID):
            continue
        matrix[token_id] = torch.from_numpy(model.get_word_vector(word))

    # Vocabulary assigns ordinary words consecutive IDs starting at 2.
    if len(vocabulary) > 2:
        matrix[vocabulary.UNK_ID] = matrix[2:].mean(dim=0)

    return nn.Embedding.from_pretrained(
        matrix, freeze=True, padding_idx=vocabulary.PAD_ID
    )


def extend_fasttext_embeddings(
    embedding: nn.Embedding,
    vocabulary: Vocabulary,
    token_sequences: Iterable[Iterable[str]],
    model_path: str | Path,
) -> tuple[nn.Embedding, Vocabulary]:
    """Append frozen FastText vectors for unseen inference words.

    Preserve every existing ID and vector, including the original UNK mean.
    The returned mapping can be reused in another call as a lookup cache.
    Only words absent from that mapping require FastText lookup. Labels and
    held-out word frequencies are not used; no parameters are optimized.
    Keep the returned embedding and vocabulary together when encoding data.
    """
    if embedding.num_embeddings != len(vocabulary):
        raise ValueError("Embedding rows must match the vocabulary")
    if embedding.embedding_dim != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-dimensional embeddings")
    if embedding.weight.requires_grad:
        raise ValueError("FastText lookup extension requires frozen embeddings")
    if embedding.padding_idx != vocabulary.PAD_ID:
        raise ValueError("Embedding padding_idx must match the vocabulary PAD ID")
    extended = vocabulary.extended(token_sequences)
    if len(extended) == len(vocabulary):
        return embedding, vocabulary

    model_path = Path(model_path).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"FastText model file not found: {model_path}")
    model = fasttext.load_model(str(model_path))
    if model.get_dimension() != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-dimensional FastText vectors")
    additional = embedding.weight.new_empty(
        (len(extended) - len(vocabulary), EMBEDDING_DIM)
    )
    for row, word in enumerate(extended.id_to_token[len(vocabulary):]):
        additional[row] = torch.as_tensor(
            model.get_word_vector(word), device=additional.device, dtype=additional.dtype
        )
    matrix = torch.cat((embedding.weight.detach(), additional), dim=0)
    return nn.Embedding.from_pretrained(
        matrix, freeze=True, padding_idx=vocabulary.PAD_ID
    ), extended
