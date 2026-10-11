"""Load all ART splits and prepare padded word-ID matrices for experiments."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from datasets import DatasetDict, load_dataset
import torch

from .preprocessing import FastTextPreprocessor
from .vocabulary import Vocabulary


DATASET_NAME = "amarfurt/art-nlp-hs26"
TEXT_FIELDS = ("observation_1", "observation_2", "hypothesis_1", "hypothesis_2")


@dataclass
class PreparedSplit:
    """Four consecutive text rows per example, plus lengths and class labels.

    token_ids: (number_of_examples * 4, longest_text_in_this_split)
    lengths: (number_of_examples * 4,), excluding padding
    labels: (number_of_examples,), with hypothesis 1 -> 0 and hypothesis 2 -> 1
    """

    token_ids: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor


@dataclass
class ExperimentData:
    """All prepared splits sharing one fixed training vocabulary."""

    train: PreparedSplit
    validation: PreparedSplit
    test: PreparedSplit
    vocabulary: Vocabulary


def load_art_dataset(cache_dir: str | None = None) -> DatasetDict:
    """Load every example from the training, validation, and test splits."""
    return load_dataset(DATASET_NAME, cache_dir=cache_dir)


def _tokenize_split(
    examples: Iterable[Mapping], processor: FastTextPreprocessor
) -> tuple[list[list[str]], list[int]]:
    """Keep the dataset's example order and TEXT_FIELDS order within each example."""
    tokens: list[list[str]] = []
    labels: list[int] = []
    for example in examples:
        label = example["label"]
        if label not in (1, 2):
            raise ValueError(f"Expected ART hypothesis label 1 or 2, got {label!r}")
        tokens.extend(processor.tokenize(example[field]) for field in TEXT_FIELDS)
        labels.append(label - 1)
    return tokens, labels


def _encode_split(
    tokens: list[list[str]], labels: list[int], vocabulary: Vocabulary
) -> PreparedSplit:
    """Right-pad with <PAD> to this split's longest text without truncating words."""
    lengths = torch.tensor([len(text) for text in tokens], dtype=torch.long)
    width = max((len(text) for text in tokens), default=0)
    token_ids = torch.full(
        (len(tokens), width), vocabulary.PAD_ID, dtype=torch.long
    )
    for row, text in enumerate(tokens):
        ids = vocabulary.encode(text)
        token_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return PreparedSplit(
        token_ids=token_ids,
        lengths=lengths,
        labels=torch.tensor(labels, dtype=torch.long),
    )


def prepare_experiment_data(dataset: Mapping[str, Iterable[Mapping]]) -> ExperimentData:
    """Preprocess all splits and encode them using a training-only vocabulary.

    Each example contributes four rows in TEXT_FIELDS order. Validation and
    test words absent from training become <UNK>. Padding widths are computed
    separately per split; no text is truncated and the vocabulary stays fixed.
    """
    processor = FastTextPreprocessor()
    train_tokens, train_labels = _tokenize_split(dataset["train"], processor)
    if not train_labels:
        raise ValueError("The training split must contain at least one example")
    vocabulary = Vocabulary(train_tokens)
    train = _encode_split(train_tokens, train_labels, vocabulary)
    del train_tokens

    validation_tokens, validation_labels = _tokenize_split(
        dataset["validation"], processor
    )
    validation = _encode_split(validation_tokens, validation_labels, vocabulary)
    del validation_tokens

    test_tokens, test_labels = _tokenize_split(dataset["test"], processor)
    test = _encode_split(test_tokens, test_labels, vocabulary)
    return ExperimentData(train, validation, test, vocabulary)


def load_experiment_data(cache_dir: str | None = None) -> ExperimentData:
    """Run the full pipeline from dataset loading to word-ID matrices."""
    return prepare_experiment_data(load_art_dataset(cache_dir=cache_dir))


def main():
    dataset = load_art_dataset()
    print(*(len(dataset[split]) for split in ("train", "validation", "test")))


if __name__ == "__main__":
    main()
