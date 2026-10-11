"""Load frozen, 300-dimensional FastText embeddings for a project vocabulary."""

from pathlib import Path

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
