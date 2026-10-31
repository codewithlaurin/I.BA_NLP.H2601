"""Retrain a selected configuration from scratch using automatic batch sizing."""

import argparse
from pathlib import Path

import torch

import train_classifier
import train_rnn
from src.art_reasoning.tracking import DEFAULT_ENTITY, DEFAULT_PROJECT


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", choices=("auto",), default="auto")
    parser.add_argument("--memory-margin", type=float, default=0.2)
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/auto-batch"))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-entity", default=DEFAULT_ENTITY)
    parser.add_argument("--wandb-group", default="P1-AutoBatch")
    parser.add_argument("--wandb-log-model", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    reference = args.reference_checkpoint.expanduser().resolve()
    if not reference.is_file():
        parser.error(f"Reference checkpoint not found: {reference}")
    checkpoint = torch.load(reference, weights_only=True, map_location="cpu")
    config = checkpoint["config"]
    model_type = checkpoint.get("model_type", "mean_pool")
    if model_type not in ("mean_pool", "rnn"):
        parser.error(f"Unsupported checkpoint model_type: {model_type}")
    inputs = (reference.parent / checkpoint["inputs_file"]).resolve()
    if not inputs.is_file():
        parser.error(f"Saved frozen inputs not found: {inputs}")
    policy = config.get("oov_policy", "unk")
    family = config.get("rnn_type", "mean_pool") if model_type == "rnn" else "mean_pool"
    # Only configuration and frozen inputs are reused. The existing entry
    # points initialize new trainable weights and optimizers from this seed.
    forwarded = [
        "--inputs-path", str(inputs), "--batch-size", args.batch_size,
        "--memory-margin", str(args.memory_margin), "--device", args.device,
        "--hidden-dim", str(config["hidden_dim"]), "--dropout", str(config["dropout"]),
        "--learning-rate", str(config["learning_rate"]), "--weight-decay", str(config["weight_decay"]),
        "--epochs", str(config["epochs"]), "--patience", str(config["patience"]),
        "--min-delta", str(config["min_delta"]), "--seed", str(args.seed),
        "--output-dir", str(args.output_dir / f"{family}-{policy}"),
        "--wandb-mode", args.wandb_mode, "--wandb-project", args.wandb_project,
        "--wandb-entity", args.wandb_entity, "--wandb-group", args.wandb_group,
        "--wandb-name", f"AutoBatch-{family}-{policy}-seed{args.seed}",
    ]
    if args.max_batch_size is not None:
        forwarded += ["--max-batch-size", str(args.max_batch_size)]
    if not args.wandb_log_model:
        forwarded += ["--no-wandb-log-model"]
    if model_type == "rnn":
        forwarded += [
            "--rnn-type", config["rnn_type"], "--rnn-hidden-size", str(config["rnn_hidden_size"]),
            "--rnn-dropout", str(config["rnn_dropout"]),
            "--bidirectional" if config.get("rnn_bidirectional", False) else "--no-bidirectional",
        ]
        if config.get("gradient_clip_norm") is None:
            forwarded += ["--no-gradient-clipping"]
        else:
            forwarded += ["--gradient-clip-norm", str(config["gradient_clip_norm"])]
    print(f"Fresh {family}/{policy} run from reference configuration {reference}; "
          f"batch sizing is automatic with {args.memory_margin:.0%} headroom.", flush=True)
    del checkpoint
    if model_type == "rnn":
        train_rnn.main(forwarded)
    else:
        train_classifier.main(forwarded)


if __name__ == "__main__":
    main()
