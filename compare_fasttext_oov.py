"""Compare UNK and FastText lookup on one saved model and a chosen held-out split."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from datasets import Dataset, DatasetInfo, load_dataset
import torch
from torch.nn import functional as F

from src.art_reasoning.batching import make_dataloader
from src.art_reasoning.data import DATASET_NAME, TEXT_FIELDS, _encode_split, _tokenize_split
from src.art_reasoning.embeddings import extend_fasttext_embeddings
from src.art_reasoning.preprocessing import FastTextPreprocessor
from src.art_reasoning.tracking import TrackingConfig, run_details, start_run
from src.art_reasoning.training import load_trained_classifier, load_trained_rnn


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@torch.no_grad()
def evaluate(model, input_builder, split, batch_size: int, device: str) -> tuple[dict, list[int]]:
    """Evaluate fixed weights, retaining ordered predictions for a paired comparison."""
    model.eval()
    if input_builder is not None:
        input_builder.eval()
    loss_sum = correct = examples = 0
    predictions = []
    for batch in make_dataloader(split, batch_size, shuffle=False):
        batch = batch.to(device)
        logits = model(batch) if input_builder is None else model(input_builder(batch))
        loss = F.cross_entropy(logits, batch.labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite evaluation loss")
        predicted = logits.argmax(dim=-1)
        count = batch.labels.numel()
        loss_sum += loss.item() * count
        correct += (predicted == batch.labels).sum().item()
        examples += count
        predictions.extend(predicted.cpu().tolist())
    if not examples:
        raise ValueError("Cannot evaluate an empty split")
    return {"loss": loss_sum / examples, "accuracy": correct / examples,
            "correct": correct, "examples": examples}, predictions


def paired_comparison(labels: list[int], original: list[int], expanded: list[int]) -> dict:
    if not (len(labels) == len(original) == len(expanded)):
        raise ValueError("Labels and predictions must have equal lengths")
    return {
        "changed_predictions": sum(a != b for a, b in zip(original, expanded)),
        "wrong_to_correct": sum(a != y and b == y for y, a, b in zip(labels, original, expanded)),
        "correct_to_wrong": sum(a == y and b != y for y, a, b in zip(labels, original, expanded)),
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Existing mean-pooling or RNN checkpoint")
    parser.add_argument("--fasttext-model", type=Path, required=True,
                        help="Original cc.en.300.bin used for training")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--data-arrow", type=Path,
                       help="Cached ART Arrow file matching --split; avoids network access")
    cache.add_argument("--validation-arrow", type=Path,
                       help="Compatibility option for the validation cache; requires --split validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oov-comparisons"))
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.validation_arrow and args.split != "validation":
        parser.error("--validation-arrow requires --split validation; use --data-arrow for test")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    fasttext_path = args.fasttext_model.expanduser().resolve()
    for path in (checkpoint_path, fasttext_path):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_type = checkpoint.get("model_type", "mean_pool")
    if model_type == "rnn":
        model, vocabulary = load_trained_rnn(checkpoint_path, "cpu")
        input_builder = None
        encoder = model.encoder
    elif model_type == "mean_pool":
        input_builder, model, vocabulary = load_trained_classifier(checkpoint_path, "cpu")
        encoder = input_builder
    else:
        parser.error(f"Unsupported checkpoint model_type: {model_type}")

    arrow_path = args.data_arrow or args.validation_arrow
    if arrow_path:
        arrow = arrow_path.expanduser().resolve()
        info = DatasetInfo.from_directory(str(arrow.parent))
        if info.dataset_name != "art-nlp-hs26" or arrow.name != f"art-nlp-hs26-{args.split}.arrow":
            parser.error(f"Arrow file must point to the cached ART {args.split} split")
        dataset = Dataset.from_file(str(arrow), info=info, split=args.split)
        if len(dataset) != info.splits[args.split].num_examples:
            raise ValueError(f"Cached {args.split} split size differs from its metadata")
        data_source = {"arrow_path": str(arrow), "arrow_sha256": sha256(arrow)}
    else:
        dataset = load_dataset(DATASET_NAME, split=args.split)
        data_source = {}
    tokens, labels = _tokenize_split(dataset, FastTextPreprocessor())
    if not labels:
        raise ValueError(f"{args.split.capitalize()} split must not be empty")
    original_split = _encode_split(tokens, labels, vocabulary)
    oov_counts = [sum(word not in vocabulary.token_to_id for word in text) for text in tokens]
    affected = [any(oov_counts[i:i + len(TEXT_FIELDS)])
                for i in range(0, len(tokens), len(TEXT_FIELDS))]
    print(f"{args.split.capitalize()}: {len(labels)} examples; replacing {sum(oov_counts)} UNK tokens "
          f"in {sum(affected)} examples. Loading FastText lookup vectors...", flush=True)
    started = perf_counter()
    expanded_embedding, expanded_vocabulary = extend_fasttext_embeddings(
        encoder.embedding, vocabulary, tokens, fasttext_path
    )
    expanded_split = _encode_split(tokens, labels, expanded_vocabulary)
    if (expanded_split.token_ids == vocabulary.UNK_ID).any().item():
        raise RuntimeError(f"FastText lookup left unknown tokens in the encoded {args.split} split")

    model.to(args.device)
    if input_builder is not None:
        input_builder.to(args.device)
    original_metrics, original_predictions = evaluate(
        model, input_builder, original_split, args.batch_size, args.device
    )
    encoder.embedding = expanded_embedding.to(args.device)
    expanded_metrics, expanded_predictions = evaluate(
        model, input_builder, expanded_split, args.batch_size, args.device
    )
    if any(a != b and not has_oov for a, b, has_oov
           in zip(original_predictions, expanded_predictions, affected)):
        raise RuntimeError("Predictions changed for examples without unknown words")

    inputs_path = (checkpoint_path.parent / checkpoint["inputs_file"]).resolve()
    extra_weights = expanded_embedding.weight[len(vocabulary):].detach().cpu().contiguous().numpy()
    result = {
        "experiment": "frozen_fasttext_oov_lookup", "phase": args.split,
        "model_type": model_type, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"],
        "inputs_sha256": sha256(inputs_path), "training_config": checkpoint["config"],
        "dataset": DATASET_NAME, "dataset_fingerprint": dataset._fingerprint,
        "dataset_version": str(dataset.info.version), **data_source,
        "fasttext_model": str(fasttext_path), "fasttext_model_bytes": fasttext_path.stat().st_size,
        "added_embedding_rows_sha256": hashlib.sha256(extra_weights.tobytes()).hexdigest(),
        "original_vocabulary_size": len(vocabulary), "expanded_vocabulary_size": len(expanded_vocabulary),
        "added_word_types": len(expanded_vocabulary) - len(vocabulary),
        "original_oov_tokens": sum(oov_counts), "tokens": sum(map(len, tokens)),
        "original_oov_fraction": sum(oov_counts) / max(1, sum(map(len, tokens))),
        "examples_with_oov": sum(affected), "expanded_oov_tokens": 0,
        "device": args.device, "batch_size": args.batch_size, "torch_version": str(torch.__version__),
        "unk": original_metrics, "fasttext": expanded_metrics,
        "accuracy_change_percentage_points": 100 * (expanded_metrics["accuracy"] - original_metrics["accuracy"]),
        "loss_change": expanded_metrics["loss"] - original_metrics["loss"],
        **paired_comparison(labels, original_predictions, expanded_predictions),
        "evaluation_duration_seconds": perf_counter() - started,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = args.output_dir.expanduser() / f"{model_type}-{args.split}-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    with (output / "predictions.jsonl").open("w") as stream:
        for index, (label, original, expanded, has_oov) in enumerate(
            zip(labels, original_predictions, expanded_predictions, affected)
        ):
            stream.write(json.dumps({"example_index": index, "label": label,
                                    "unk_prediction": original, "fasttext_prediction": expanded,
                                    "has_oov": has_oov}) + "\n")
    print(json.dumps({key: result[key] for key in (
        "unk", "fasttext", "added_word_types", "original_oov_fraction", "examples_with_oov",
        "accuracy_change_percentage_points", "loss_change", "wrong_to_correct", "correct_to_wrong",
    )}, indent=2), flush=True)
    print(f"Saved comparison: {summary_path}", flush=True)
    if args.wandb_mode != "disabled":
        metadata = {key: result[key] for key in (
            "experiment", "phase", "model_type", "checkpoint", "checkpoint_sha256",
            "checkpoint_epoch", "inputs_sha256", "training_config", "dataset",
            "dataset_fingerprint", "fasttext_model", "added_embedding_rows_sha256", "device", "batch_size",
        )}
        with start_run(metadata, f"OOV-{args.split}-{checkpoint_path.parent.parent.name}", output,
                       TrackingConfig(mode=args.wandb_mode, log_model=False),
                       group="P1-FastText-OOV" + ("-Test" if args.split == "test" else ""),
                       job_type="evaluate") as run:
            metrics = {f"{args.split}/{policy}_{metric}": value
                       for policy in ("unk", "fasttext")
                       for metric, value in result[policy].items()}
            metrics.update({key: result[key] for key in (
                "accuracy_change_percentage_points", "loss_change", "wrong_to_correct", "correct_to_wrong",
            )})
            run.log(metrics)
            run.summary.update(metrics)
            result.update(run_details(run, args.wandb_mode))
        summary_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
