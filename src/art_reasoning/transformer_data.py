"""Prepare full ART candidate sequences and dynamically padded BERT batches."""

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .data import TEXT_FIELDS


class CandidateDataset(Dataset):
    """Keep two unpadded candidates and one zero-based label per example."""

    def __init__(
        self, examples, tokenizer: PreTrainedTokenizerBase, max_length: int = 512
    ) -> None:
        self.examples = []
        for index, example in enumerate(examples):
            if example["label"] not in (1, 2):
                raise ValueError(f"Example {index}: expected label 1 or 2")
            if any(
                not isinstance(example[field], str) or not example[field].strip()
                for field in TEXT_FIELDS
            ):
                raise ValueError(f"Example {index}: text fields must be nonempty strings")
            encoded = tokenizer(
                [example["observation_1"]] * 2,
                [example[f"hypothesis_{choice}"] + " " + example["observation_2"]
                 for choice in (1, 2)],
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            if any(len(ids) > max_length for ids in encoded["input_ids"]):
                raise ValueError(
                    f"Example {index}: candidate exceeds {max_length} tokens; "
                    "inputs are never truncated"
                )
            candidates = [
                {name: values[choice] for name, values in encoded.items()}
                for choice in range(2)
            ]
            self.examples.append((candidates, example["label"] - 1))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


@dataclass
class CandidateBatch:
    """Token tensors shaped (B, 2, L) and labels shaped (B,)."""

    inputs: dict[str, torch.Tensor]
    labels: torch.Tensor

    def to(self, device: torch.device | str) -> "CandidateBatch":
        return CandidateBatch(
            {name: tensor.to(device) for name, tensor in self.inputs.items()},
            self.labels.to(device),
        )


@dataclass
class CandidateCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, examples: list) -> CandidateBatch:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        if self.tokenizer.padding_side != "right":
            raise ValueError("BERT candidate scoring requires right padding")
        candidates = [candidate for choices, _ in examples for candidate in choices]
        padded = self.tokenizer.pad(candidates, padding=True, return_tensors="pt")
        return CandidateBatch(
            {name: tensor.reshape(len(examples), 2, -1)
             for name, tensor in padded.items()},
            torch.tensor([label for _, label in examples], dtype=torch.long),
        )
