"""Evaluate a saved ART Transformer checkpoint without training or W&B logging."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BertConfig, BertModel

from src.art_reasoning.batching import select_device
from src.art_reasoning.data import DATASET_NAME
from src.art_reasoning.models.bert_candidate_scorer import BertCandidateScorer
from src.art_reasoning.transformer_data import CandidateCollator, CandidateDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    run_dir = args.checkpoint.parent
    device = select_device() if args.device == "auto" else torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer", local_files_only=True)
    model = BertCandidateScorer(BertModel(
        BertConfig.from_pretrained(run_dir / "encoder_config"), add_pooling_layer=False,
    ))
    model.load_state_dict(saved["model_state"])
    model.to(device).eval()
    raw = load_dataset(DATASET_NAME, split=args.split)
    data = CandidateDataset(raw, tokenizer, model.encoder.config.max_position_embeddings)
    loader = DataLoader(data, batch_size=args.batch_size, collate_fn=CandidateCollator(tokenizer))
    confusion = torch.zeros(2, 2, dtype=torch.long)
    loss_sum = correct = count = 0
    predictions = []
    started = perf_counter()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluate {args.split}"):
            batch = batch.to(device)
            logits = model(batch)
            loss_sum += torch.nn.functional.cross_entropy(logits, batch.labels, reduction="sum").item()
            probabilities = logits.softmax(dim=-1).cpu().tolist()
            predicted = logits.argmax(dim=-1).cpu().tolist()
            labels = batch.labels.cpu().tolist()
            for scores, probabilities_i, prediction, label in zip(
                logits.cpu().tolist(), probabilities, predicted, labels, strict=True
            ):
                confusion[label, prediction] += 1
                correct += prediction == label
                predictions.append({
                    "example_index": count, **dict(raw[count]),
                    "predicted_label": prediction + 1,
                    "logits": scores, "probabilities": probabilities_i,
                })
                count += 1
    if count != len(raw) or count == 0:
        raise RuntimeError("Evaluation must include every example exactly once")
    with args.checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    metrics = {
        "split": args.split, "examples": count, "correct": correct,
        "accuracy": correct / count, "loss": loss_sum / count,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_labels": [1, 2], "confusion_matrix_axes": "rows=true, columns=predicted",
        "checkpoint": args.checkpoint.name, "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": saved["epoch"], "dataset_fingerprint": raw._fingerprint,
        "device": str(device), "batch_size": args.batch_size,
        "max_candidate_tokens": max(len(c["input_ids"]) for choices, _ in data for c in choices),
        "truncation": False, "wandb": "disabled",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "inference_seconds": perf_counter() - started,
    }
    (run_dir / f"{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (run_dir / f"{args.split}_predictions.jsonl").open("w") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
