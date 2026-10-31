"""Prepare ART texts or train with a FastText binary or saved frozen inputs."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4

import torch

from src.art_reasoning.batching import find_gpu_batch_size, make_dataloader, select_device
from src.art_reasoning.data import DATASET_NAME, TEXT_FIELDS, load_experiment_data
from src.art_reasoning.embeddings import load_fasttext_embeddings
from src.art_reasoning.models.classifier import Classifier
from src.art_reasoning.models.classifier_input import ClassifierInput
from src.art_reasoning.tracking import (
    DEFAULT_ENTITY, DEFAULT_PROJECT, TrackingConfig, run_details, start_run,
)
from src.art_reasoning.training import (
    TrainingConfig, build_optimizer, load_inputs, load_trained_classifier, run_epoch,
    reproducibility_metadata, save_inputs, seed_everything, train_classifier,
)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", default="auto", help="Examples per batch, or auto (default)")
    parser.add_argument("--memory-margin", type=float, default=0.2, help="Free GPU memory fraction to reserve")
    parser.add_argument("--max-batch-size", type=int, help="Upper limit for auto sizing")
    parser.add_argument("--cache-dir", help="Optional Hugging Face dataset cache directory")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fasttext-model", type=Path, help="Local 300d FastText .bin file; enables training")
    source.add_argument("--inputs-path", type=Path, help="Reuse a saved frozen matrix and vocabulary; enables training")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--hidden-dim", type=int, default=128)
    learning_rates = parser.add_mutually_exclusive_group()
    learning_rates.add_argument("--learning-rate", type=float, default=1e-3)
    learning_rates.add_argument("--learning-rates", type=float, nargs="+", help="Compare rates using validation loss")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs per trial")
    parser.add_argument("--patience", type=int, default=5, help="Validation checks without improvement before stopping")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum loss decrease to reset patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate the selected best checkpoint on test once")
    parser.add_argument(
        "--wandb-name", default=os.environ.get("WANDB_NAME"),
        help="Run name or template with {learning_rate} and {seed} for multi-rate runs",
    )
    parser.add_argument(
        "--wandb-group", default=os.environ.get("WANDB_RUN_GROUP"),
        help="Shared W&B group (default: unique experiment directory name)",
    )
    parser.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
        help="W&B project (default: WANDB_PROJECT or nlp-classifier)",
    )
    parser.add_argument(
        "--wandb-entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY),
        help="W&B username/team (default: WANDB_ENTITY or laurin-uni-workspace)",
    )
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"), help="W&B mode (default: online)",
    )
    parser.add_argument(
        "--wandb-log-model", action=argparse.BooleanOptionalAction, default=True,
        help="Log each trial's best checkpoint and frozen inputs (default: enabled)",
    )
    args = parser.parse_args(argv)
    rates = args.learning_rates or [args.learning_rate]
    try:
        tracking_config = TrackingConfig(
            project=args.wandb_project, entity=args.wandb_entity,
            mode=args.wandb_mode, log_model=args.wandb_log_model,
        )
        configurations = [TrainingConfig(
            learning_rate=rate, weight_decay=args.weight_decay, dropout=args.dropout,
            hidden_dim=args.hidden_dim, epochs=args.epochs, patience=args.patience,
            min_delta=args.min_delta, seed=args.seed,
        ) for rate in rates]
        run_names = [
            args.wandb_name.format(learning_rate=f"{config.learning_rate:g}", seed=config.seed)
            if args.wandb_name else None
            for config in configurations
        ]
    except (ValueError, KeyError, IndexError) as error:
        parser.error(str(error))
    if not 0 <= args.memory_margin < 1:
        parser.error("--memory-margin must be between 0 (inclusive) and 1")
    if args.max_batch_size is not None and args.max_batch_size < 1:
        parser.error("--max-batch-size must be positive")
    source_path = args.inputs_path or args.fasttext_model
    if source_path is not None and not source_path.expanduser().is_file():
        parser.error(f"Input source not found: {source_path}")
    if args.evaluate_test and source_path is None:
        parser.error("--evaluate-test requires --fasttext-model or --inputs-path")
    batch_size = None
    if args.batch_size != "auto":
        try:
            batch_size = int(args.batch_size)
        except ValueError:
            parser.error("--batch-size must be a positive integer or auto")
        if batch_size < 1:
            parser.error("--batch-size must be positive")

    embedding = vocabulary = None
    if args.inputs_path is not None:
        embedding, vocabulary = load_inputs(args.inputs_path)
    data = load_experiment_data(cache_dir=args.cache_dir, include_test=args.evaluate_test,
                                vocabulary=vocabulary)
    print(f"Vocabulary size (including <PAD> and <UNK>): {len(data.vocabulary)}")
    print(f"Text rows within each example: {', '.join(TEXT_FIELDS)}")
    splits = {name: getattr(data, name) for name in ("train", "validation")}
    if data.test is not None:
        splits["test"] = data.test
    for name, split in splits.items():
        print(f"{name}: {len(split)} examples, ID matrix {tuple(split.token_ids.shape)}")
    device = select_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is unavailable")
    input_builder = None
    probe_step = None
    if source_path is not None:
        if embedding is None:
            embedding = load_fasttext_embeddings(data.vocabulary, args.fasttext_model)
        input_builder = ClassifierInput(embedding).to(device).eval()

        def probe_step(batch):
            # Recreate the disposable head/optimizer per attempt: an OOM must
            # neither change the actual model nor leave partial AdamW state.
            head = Classifier(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device).train()
            optimizer = build_optimizer(head, configurations[0])
            try:
                loss = torch.nn.functional.cross_entropy(head(input_builder(batch)), batch.labels)
                loss.backward()
                optimizer.step()
                return head, optimizer, loss.detach()
            finally:
                optimizer.zero_grad(set_to_none=True)

    if batch_size is None:
        # Only lengths from held-out data affect capacity; never vocabulary or tuning.
        longest_split = max(splits.values(), key=lambda split: split.token_ids.shape[1])
        batch_size = find_gpu_batch_size(
            longest_split, device, memory_margin=args.memory_margin,
            max_batch_size=min(args.max_batch_size or len(data.train), len(data.train)),
            probe_step=probe_step,
        )
    print(f"Device: {device}; batch size: {batch_size} complete examples")
    if input_builder is None:
        batch = next(iter(make_dataloader(data.train, batch_size, shuffle=True, seed=args.seed)))
        print(f"First training batch: IDs {tuple(batch.token_ids.shape)}")
        print(f"Original lengths for its first example: {batch.lengths[0].tolist()}")
        print("Data prepared. Supply --fasttext-model or --inputs-path to train the classifier.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    experiment_dir = args.output_dir.expanduser() / f"{stamp}-{uuid4().hex[:8]}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    inputs_path = experiment_dir / "inputs.pt"
    save_inputs(input_builder, data.vocabulary, inputs_path)
    versions = reproducibility_metadata(input_builder, data.vocabulary)
    print(
        f"Experiment: {experiment_dir}; W&B {tracking_config.mode} "
        f"({tracking_config.entity}/{tracking_config.project}); local metrics enabled",
        flush=True,
    )
    results = []
    for index, config in enumerate(configurations, start=1):
        seed_everything(config.seed)
        classifier = Classifier(hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)
        train_loader = make_dataloader(data.train, batch_size, shuffle=True, seed=config.seed)
        validation_loader = make_dataloader(data.validation, batch_size, seed=config.seed)
        result = train_classifier(
            input_builder, classifier, train_loader, validation_loader, device, config,
            experiment_dir / f"trial-{index}-lr-{config.learning_rate:g}", inputs_path,
            metadata={
                "dataset": DATASET_NAME,
                "fasttext_model": str(args.fasttext_model) if args.fasttext_model else None,
                "frozen_inputs_source": str(args.inputs_path) if args.inputs_path else None,
                "torch_version": str(torch.__version__), "vocabulary_size": len(data.vocabulary),
                "split_sizes": {name: len(split) for name, split in splits.items()},
                "device": str(device), "memory_margin": args.memory_margin,
                "batch_size_automatic": args.batch_size == "auto",
                "dataset_split_versions": data.split_versions,
                **versions,
            },
            tracking_config=tracking_config,
            tracking_group=args.wandb_group or experiment_dir.name,
            tracking_name=run_names[index - 1],
        )
        results.append(result)
        del classifier
    selected = min(results, key=lambda result: result["best_validation_loss"])
    summary = {"trials": results, "selected": selected}
    if args.evaluate_test:
        # Free the current embedding before restoring the chosen checkpoint's copy.
        del input_builder, embedding
        input_builder, classifier, _ = load_trained_classifier(selected["best_checkpoint"], device)
        # Keep final test evaluation separate from the training trials. In
        # offline mode finished W&B runs cannot be resumed to append metrics.
        with start_run(
            {"selected": selected, "dataset": DATASET_NAME, "wandb": asdict(tracking_config)},
            "selected-model-test", experiment_dir / "evaluation", tracking_config,
            group=args.wandb_group or experiment_dir.name, job_type="evaluate",
        ) as tracking:
            summary["test"] = run_epoch(
                input_builder, classifier, make_dataloader(data.test, batch_size), device
            )
            tracking.log({f"test/{key}": value for key, value in summary["test"].items()})
            tracking.summary.update({"selected": selected})
            summary["evaluation_tracking"] = run_details(tracking, tracking_config.mode)
        print(f"Selected model test metrics: {summary['test']}")
    (experiment_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Best checkpoint: {selected['best_checkpoint']}")


if __name__ == "__main__":
    main()
