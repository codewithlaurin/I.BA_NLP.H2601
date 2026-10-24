"""Prepare all ART texts and dynamically padded batches for experiments."""

import argparse
from pathlib import Path

import torch

from src.art_reasoning.batching import (
    find_gpu_batch_size,
    make_dataloader,
    select_device,
)
from src.art_reasoning.data import TEXT_FIELDS, load_experiment_data
from src.art_reasoning.embeddings import load_fasttext_embeddings
from src.art_reasoning.models.classifier import Classifier
from src.art_reasoning.models.classifier_input import ClassifierInput


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size", default="auto", help="Examples per batch, or auto (default)"
    )
    parser.add_argument(
        "--memory-margin", type=float, default=0.2,
        help="Fraction of free GPU memory to reserve (default: 0.2)",
    )
    parser.add_argument("--max-batch-size", type=int, help="Upper limit for auto sizing")
    parser.add_argument("--cache-dir", help="Optional Hugging Face dataset cache directory")
    parser.add_argument(
        "--fasttext-model", type=Path,
        help="Local 300d FastText .bin file; build features and run the classifier",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128,
        help="Units in the classifier's first layer (default: 128)",
    )
    args = parser.parse_args()
    if not 0 <= args.memory_margin < 1:
        parser.error("--memory-margin must be between 0 (inclusive) and 1")
    if args.max_batch_size is not None and args.max_batch_size < 1:
        parser.error("--max-batch-size must be positive")
    if args.fasttext_model is not None and not args.fasttext_model.expanduser().is_file():
        parser.error(f"FastText model file not found: {args.fasttext_model}")
    if args.hidden_dim < 1:
        parser.error("--hidden-dim must be positive")
    batch_size = None
    if args.batch_size != "auto":
        try:
            batch_size = int(args.batch_size)
        except ValueError:
            parser.error("--batch-size must be a positive integer or auto")
        if batch_size < 1:
            parser.error("--batch-size must be positive")

    data = load_experiment_data(cache_dir=args.cache_dir)
    print(f"Vocabulary size (including <PAD> and <UNK>): {len(data.vocabulary)}")
    print(f"Text rows within each example: {', '.join(TEXT_FIELDS)}")
    splits = {name: getattr(data, name) for name in ("train", "validation", "test")}
    for name, split in splits.items():
        print(
            f"{name}: {len(split.labels)} examples, "
            f"ID matrix {tuple(split.token_ids.shape)}, "
            f"lengths {tuple(split.lengths.shape)}, "
            f"labels {tuple(split.labels.shape)}"
        )
    device = select_device()
    input_builder = None
    classifier = None
    probe_step = None
    if args.fasttext_model is not None:
        embedding = load_fasttext_embeddings(data.vocabulary, args.fasttext_model)
        input_builder = ClassifierInput(embedding).to(device).eval()
        classifier = Classifier(hidden_dim=args.hidden_dim).to(device).eval()
        probe_step = lambda probe_batch: classifier(input_builder(probe_batch))
    if batch_size is None:
        # Include the longest text from any split so evaluation batches also fit.
        longest_split = max(splits.values(), key=lambda split: split.token_ids.shape[1])
        batch_size = find_gpu_batch_size(
            longest_split,
            device,
            memory_margin=args.memory_margin,
            max_batch_size=min(args.max_batch_size or len(data.train), len(data.train)),
            probe_step=probe_step,
        )
        if device.type != "cpu":
            print(
                "Auto sizing covers the available forward pass; backward and "
                "optimizer memory must be included when training is added."
            )
    print(f"Device: {device}; batch size: {batch_size} complete examples")
    loaders = {
        name: make_dataloader(split, batch_size, shuffle=(name == "train"))
        for name, split in splits.items()
    }
    for name, loader in loaders.items():
        print(f"{name}: {len(loader)} batches (including the final partial batch)")
    batch = next(iter(loaders["train"])).to(device)
    print(f"First training batch: IDs {tuple(batch.token_ids.shape)}")
    print(f"Original lengths for its first example: {batch.lengths[0].tolist()}")
    if input_builder is not None:
        features = input_builder(batch)
        print(f"Classifier input: {tuple(features.shape)} ({features.dtype})")
        print(f"Class labels: {tuple(batch.labels.shape)}")
        with torch.no_grad():
            logits = classifier(features)
            probabilities = classifier.output_activation(logits)
            predictions = probabilities.argmax(dim=-1)
        print(f"Classifier: {classifier}")
        print(f"Classifier output logits: {tuple(logits.shape)} (model is untrained)")
        print(f"Output probabilities: {tuple(probabilities.shape)}")
        print(f"First example's hypothesis probabilities: {probabilities[0].tolist()}")
        print(f"First example's selected hypothesis: {int(predictions[0]) + 1} (untrained)")


if __name__ == "__main__":
    main()
