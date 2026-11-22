"""Train ART candidate scoring with a pretrained or randomly initialized BERT encoder."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertConfig, BertModel, get_linear_schedule_with_warmup

from src.art_reasoning.batching import find_largest_batch_size, select_device
from src.art_reasoning.data import load_art_dataset
from src.art_reasoning.models.bert_candidate_scorer import (
    BertCandidateScorer,
    load_bert_components,
)
from src.art_reasoning.tracking import TrackingConfig, run_details, start_run
from src.art_reasoning.training import _save_checkpoint, run_epoch, seed_everything
from src.art_reasoning.transformer_data import CandidateCollator, CandidateDataset

CHECKPOINT = "google-bert/bert-base-cased"
REVISION = "cd5ef92a9fb2f889e972770a36d4ed042daf221e"
# Settings that must match when resuming, because they define the run.
# The batch size is checked separately because it may be chosen automatically.
RESUME_SETTINGS = (
    "initialization", "epochs", "learning_rate", "weight_decay", "warmup_ratio", "seed",
)


def parse_batch_size(value: str) -> str | int:
    """Accept 'auto' or a positive number of examples per batch."""
    if value == "auto":
        return value
    size = int(value)
    if size < 1:
        raise argparse.ArgumentTypeError("batch size must be positive")
    return size


def find_max_batch_size(
    train: CandidateDataset,
    collator: CandidateCollator,
    encoder_config: BertConfig,
    device: torch.device,
    memory_margin: float,
    max_batch_size: int | None,
) -> int:
    """Largest training batch for which a full AdamW step fits on the GPU.

    Every probe batch repeats the training example with the longest candidate,
    so no real batch can be longer. A disposable model with the same
    architecture is measured, while the real model waits on the CPU.
    """
    longest = max(train, key=lambda item: max(len(c["input_ids"]) for c in item[0]))
    disposable = BertCandidateScorer(BertModel(encoder_config, add_pooling_layer=False)).to(device)
    optimizer = torch.optim.AdamW(disposable.parameters())

    def make_batch(size: int):
        return collator([longest] * size).to(device)

    def training_step(batch) -> None:
        loss = torch.nn.functional.cross_entropy(disposable(batch), batch.labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(disposable.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    limit = len(train) if max_batch_size is None else min(max_batch_size, len(train))
    return find_largest_batch_size(
        make_batch, device, limit, memory_margin=memory_margin, probe_step=training_step
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--epochs", type=int, default=10, help="Epoch cap; also the schedule length")
    parser.add_argument(
        "--batch-size", type=parse_batch_size, default="auto",
        help="Examples per batch, or 'auto' for the largest that fits on the GPU",
    )
    parser.add_argument("--memory-margin", type=float, default=0.2,
                        help="Share of free GPU memory kept unused by --batch-size auto")
    parser.add_argument("--max-batch-size", type=int, help="Optional cap for --batch-size auto")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Share of steps for linear warmup")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume an epoch-boundary last.pt checkpoint")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-project", default="nlp-transformers")
    parser.add_argument("--wandb-group")
    args = parser.parse_args()
    if min(args.epochs, args.patience) < 1:
        parser.error("epochs and patience must be positive")
    if args.max_batch_size is not None and args.max_batch_size < 1:
        parser.error("max batch size must be positive")
    if not 0 <= args.memory_margin < 1:
        parser.error("memory margin must be in [0, 1)")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay nonnegative")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("warmup ratio must be in [0, 1)")
    seed_everything(args.seed)
    device = select_device() if args.device == "auto" else torch.device(args.device)
    run_name = f"{args.initialization}-lr{args.learning_rate:g}-seed{args.seed}"
    resumed = None
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=True)
        for name in RESUME_SETTINGS:
            if getattr(args, name) != resumed["settings"][name]:
                parser.error(f"Resume requires the original --{name.replace('_', '-')}")
        if args.batch_size not in ("auto", resumed["settings"]["batch_size"]):
            parser.error("Resume requires the original --batch-size (or auto)")
    elif args.batch_size == "auto" and device.type not in ("cuda", "mps"):
        parser.error("--batch-size auto needs a CUDA or MPS GPU; pass a number on CPU")
    output_dir = args.resume.parent if args.resume else args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("outputs/project2") / f"{run_name}-{timestamp}"
    if not args.resume:
        output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Device: {device}; output: {output_dir}", flush=True)
    model, tokenizer = load_bert_components(
        pretrained=args.initialization == "pretrained" and resumed is None,
        checkpoint=CHECKPOINT, revision=REVISION,
    )
    if resumed:
        model.load_state_dict(resumed["model_state"])
    tokenizer.save_pretrained(output_dir / "tokenizer")
    model.encoder.config.save_pretrained(output_dir / "encoder_config")
    raw = load_art_dataset(include_test=False)
    collator = CandidateCollator(tokenizer)
    splits = {
        name: CandidateDataset(raw[name], tokenizer, model.encoder.config.max_position_embeddings)
        for name in ("train", "validation")
    }
    if resumed:
        batch_size = resumed["settings"]["batch_size"]
        batch_size_selection = resumed["settings"]["batch_size_selection"]
    elif args.batch_size == "auto":
        print("Probing the largest training batch that fits on the GPU...", flush=True)
        batch_size = find_max_batch_size(
            splits["train"], collator, model.encoder.config, device,
            args.memory_margin, args.max_batch_size,
        )
        batch_size_selection = "auto"
    else:
        batch_size, batch_size_selection = args.batch_size, "fixed"
    print(f"Batch size: {batch_size} examples ({batch_size_selection})", flush=True)
    model.to(device)
    validation_loader = DataLoader(splits["validation"], batch_size=batch_size, collate_fn=collator)
    steps_per_epoch = math.ceil(len(splits["train"]) / batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = round(args.warmup_ratio * total_steps)
    settings = {
        **{name: value for name, value in vars(args).items()
           if name not in ("output_dir", "resume", "batch_size")},
        "batch_size": batch_size, "batch_size_selection": batch_size_selection,
        "run_name": run_name, "checkpoint": CHECKPOINT, "revision": REVISION, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "optimizer": "AdamW", "loss": "CrossEntropyLoss", "gradient_clip_norm": 1.0,
        "scheduler": "linear_warmup_decay", "warmup_steps": warmup_steps, "total_steps": total_steps,
        "early_stopping_metric": "validation accuracy, validation loss as tie-breaker",
        "truncation": False, "padding": "dynamic",
        "split_fingerprints": {name: raw[name]._fingerprint for name in splits},
    }
    (output_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if parameter.ndim == 1 else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.learning_rate,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if resumed:
        # Load after creating the scheduler: the saved optimizer state restores the
        # current learning rate, and the scheduler state restores the step count.
        optimizer.load_state_dict(resumed["optimizer_state"])
        scheduler.load_state_dict(resumed["scheduler_state"])
    best = tuple(resumed["best"]) if resumed else (-1.0, float("-inf"))
    stale = resumed["stale"] if resumed else 0
    start_epoch = resumed["epoch"] + 1 if resumed else 1
    last_epoch = start_epoch - 1
    # Runtime continues across resumes so it reports the total training time.
    previous_seconds = resumed["metrics"]["elapsed_seconds"] if resumed else 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tracking_config = TrackingConfig(project=args.wandb_project, mode=args.wandb_mode, log_model=False)
    with start_run(
        settings, run_name, output_dir, tracking_config, args.wandb_group,
        run_id=resumed["wandb_run_id"] if resumed else None,
    ) as tracking:
        identity = run_details(tracking, args.wandb_mode)
        (output_dir / "wandb.json").write_text(json.dumps(identity, indent=2) + "\n")
        started = perf_counter()
        for epoch in range(start_epoch, args.epochs + 1):
            # Epoch-specific seeds make dropout and shuffling reproducible on resume.
            seed_everything(args.seed + epoch)
            train_loader = DataLoader(
                splits["train"], batch_size=batch_size, shuffle=True, collate_fn=collator,
                generator=torch.Generator().manual_seed(args.seed + epoch),
            )
            train = run_epoch(
                None, model, tqdm(train_loader, desc=f"Epoch {epoch} train"), device,
                optimizer, 1.0, scheduler,
            )
            validation = run_epoch(None, model, tqdm(validation_loader, desc=f"Epoch {epoch} validation"), device)
            metrics = {
                "epoch": epoch, "train": train, "validation": validation,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": previous_seconds + perf_counter() - started,
            }
            with (output_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            tracking.log({
                "epoch": epoch, "learning_rate": metrics["learning_rate"],
                "elapsed_seconds": metrics["elapsed_seconds"],
                **{f"train/{key}": value for key, value in train.items() if key != "examples"},
                **{f"validation/{key}": value for key, value in validation.items() if key != "examples"},
            }, step=epoch)
            last_epoch = epoch
            score = (validation["accuracy"], -validation["loss"])
            payload = {"model_state": model.state_dict(), "epoch": epoch, "metrics": metrics}
            if score > best:
                best, stale = score, 0
                _save_checkpoint(payload, output_dir / "best.pt")
            else:
                stale += 1
            _save_checkpoint(
                {**payload, "optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict(), "settings": settings,
                 "best": best, "stale": stale, "wandb_run_id": identity["wandb_run_id"]},
                output_dir / "last.pt",
            )
            print(json.dumps(metrics), flush=True)
            if stale >= args.patience:
                break
        training_seconds = previous_seconds + perf_counter() - started
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
        # Verify that saved weights restore the best validation result without the source weights.
        saved = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
        del scheduler, optimizer, model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        restored = BertCandidateScorer(BertModel(
            BertConfig.from_pretrained(output_dir / "encoder_config"), add_pooling_layer=False,
        ))
        restored.load_state_dict(saved["model_state"])
        restored.to(device)
        validation = run_epoch(None, restored, tqdm(validation_loader, desc="Restore check"), device)
        expected = saved["metrics"]["validation"]
        if validation["accuracy"] != expected["accuracy"] or abs(validation["loss"] - expected["loss"]) > 1e-5:
            raise RuntimeError("Restored checkpoint does not reproduce validation metrics")
        summary = {
            "best_epoch": saved["epoch"],
            "best_validation_accuracy": validation["accuracy"],
            "best_validation_loss": validation["loss"],
            "epochs_completed": last_epoch,
            "stopped_early": last_epoch < args.epochs,
            "checkpoint_verified": True,
            "training_seconds": training_seconds,
            "peak_gpu_memory_gb": peak_memory,
            **identity,
        }
        tracking.summary.update(summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # The run is complete, so the resume checkpoint (weights + optimizer, ~1.3 GB) is no longer needed.
    (output_dir / "last.pt").unlink(missing_ok=True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
