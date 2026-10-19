"""Pad complete examples in batches and probe GPU batch capacity."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .data import PreparedSplit, TEXT_FIELDS
from .vocabulary import Vocabulary


@dataclass
class TextBatch:
    """IDs shaped (B, 4, L), original lengths (B, 4), and labels (B,)."""

    token_ids: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor

    @property
    def mask(self) -> torch.Tensor:
        """True for original tokens and False for padding."""
        positions = torch.arange(self.token_ids.shape[-1], device=self.token_ids.device)
        return positions < self.lengths.unsqueeze(-1)

    def to(self, device: torch.device | str) -> "TextBatch":
        return TextBatch(
            self.token_ids.to(device), self.lengths.to(device), self.labels.to(device)
        )


def collate_text_batch(examples: list) -> TextBatch:
    """Right-pad all four fields to the batch maximum; never truncate a text."""
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    sequences = [sequence for fields, _ in examples for sequence in fields]
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    ids = pad_sequence(sequences, batch_first=True, padding_value=Vocabulary.PAD_ID)
    # Keep a usable sequence axis even when every text in the batch is empty.
    if ids.shape[1] == 0:
        ids = torch.full((len(sequences), 1), Vocabulary.PAD_ID, dtype=torch.long)
    return TextBatch(
        token_ids=ids.reshape(len(examples), len(TEXT_FIELDS), -1),
        lengths=lengths.reshape(len(examples), len(TEXT_FIELDS)),
        labels=torch.stack([label for _, label in examples]),
    )


def make_dataloader(
    split: PreparedSplit, batch_size: int, *, shuffle: bool = False, seed: int = 42
) -> DataLoader:
    """Batch by complete example, preserving all examples including the last batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return DataLoader(
        split,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_text_batch,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )


def select_device() -> torch.device:
    """Prefer CUDA, then Apple's MPS GPU, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _largest_fitting_size(fits: Callable[[int], bool], limit: int) -> int:
    """Find the largest successful size by doubling, then binary search."""
    if not fits(1):
        raise RuntimeError("One complete example does not fit with the memory margin")
    low = 1
    while low < limit:
        high = min(low * 2, limit)
        if not fits(high):
            break
        low = high
    else:
        return low
    while low + 1 < high:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low


def find_gpu_batch_size(
    split: PreparedSplit,
    device: torch.device | str,
    *,
    memory_margin: float = 0.2,
    max_batch_size: int | None = None,
    cpu_batch_size: int = 64,
    probe_step: Callable[[TextBatch], object] | None = None,
) -> int:
    """Probe worst-length batches with headroom and recover from GPU OOM.

    Without probe_step, capacity covers input tensors only. For training,
    load the model on the GPU first and supply a callback that runs a full
    forward/backward/optimizer step on a disposable model and optimizer.
    The callback must clear its gradients and not retain batches or graphs;
    its returned object remains alive until memory is measured.

    CUDA uses peak reserved memory and reserves memory_margin of initially
    free VRAM. MPS checks driver memory against its recommended working set
    and additionally reduces the successful batch size by memory_margin,
    since MPS does not expose the same peak-memory statistic. CPU uses the
    configurable fallback. No sequence is shortened to make a batch fit.
    """
    device = torch.device(device)
    if not 0 <= memory_margin < 1:
        raise ValueError("memory_margin must be between 0 (inclusive) and 1")
    if not len(split):
        raise ValueError("Cannot select a batch size for an empty split")
    limit = len(split) if max_batch_size is None else max_batch_size
    if limit < 1 or cpu_batch_size < 1:
        raise ValueError("Batch size limits must be positive")
    if device.type == "cpu":
        return min(cpu_batch_size, limit)
    if device.type not in ("cuda", "mps"):
        raise ValueError("Automatic sizing supports CUDA, MPS, or CPU")

    # Repeat the longest original text in every field to probe the largest
    # possible padded/packed workload, rather than a randomly short batch.
    row = int(split.lengths.argmax())
    length = int(split.lengths[row])
    width = max(1, length)
    sequence = split.token_ids[row, :width]
    if sequence.numel() == 0:
        sequence = torch.full((1,), Vocabulary.PAD_ID, dtype=torch.long)

    backend = torch.cuda if device.type == "cuda" else torch.mps
    backend.empty_cache()
    if device.type == "cuda":
        free_memory, _ = torch.cuda.mem_get_info(device)
        baseline = torch.cuda.memory_reserved(device)
    else:
        baseline = torch.mps.driver_allocated_memory()
        free_memory = max(0, torch.mps.recommended_max_memory() - baseline)
    budget = free_memory * (1 - memory_margin)

    def fits(size: int) -> bool:
        ids = batch = result = None
        try:
            backend.empty_cache()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            ids = sequence.to(device).reshape(1, 1, width).expand(
                size, len(TEXT_FIELDS), width
            ).clone()
            batch = TextBatch(
                token_ids=ids,
                lengths=torch.full(
                    (size, len(TEXT_FIELDS)), length, dtype=torch.long, device=device
                ),
                labels=torch.zeros(size, dtype=torch.long, device=device),
            )
            ids = None
            if probe_step is not None:
                result = probe_step(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                used = torch.cuda.max_memory_reserved(device) - baseline
            else:
                torch.mps.synchronize()
                used = torch.mps.driver_allocated_memory() - baseline
            return used <= budget
        except RuntimeError as error:
            if isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower():
                return False
            raise
        finally:
            ids = batch = result = None
            backend.empty_cache()

    largest = _largest_fitting_size(fits, limit)
    if device.type == "mps":
        largest = max(1, int(largest * (1 - memory_margin)))
    # Recheck the selected size after cleanup, since free GPU memory can change.
    if not fits(largest):
        raise RuntimeError("Available GPU memory changed during batch-size selection")
    return largest
