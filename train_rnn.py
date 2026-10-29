"""Train frozen-FastText ART models with a two-layer GRU or LSTM and classifier."""

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
from src.art_reasoning.tracking import (
    DEFAULT_ENTITY, DEFAULT_PROJECT, TrackingConfig, run_details, start_run,
)
from src.art_reasoning.training import (
    RNNTrainingConfig, build_optimizer, build_rnn_model, load_inputs, load_trained_rnn,
    reproducibility_metadata, run_epoch, save_inputs, seed_everything, train_classifier,
)


DEFAULT_GROUP = "P1-RNN"


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fasttext-model", type=Path, help="Local 300d FastText .bin model")
    source.add_argument("--inputs-path", type=Path, help="Reuse a previous experiment's inputs.pt")
    parser.add_argument("--rnn-type", choices=("gru", "lstm"), default="gru")
    parser.add_argument("--rnn-hidden-size", type=int, default=128, help="Hidden width per direction")
    parser.add_argument(
        "--bidirectional", action=argparse.BooleanOptionalAction, default=False,
        help="Encode each text forward and backward (default: forward-only)",
    )
    parser.add_argument("--rnn-dropout", type=float, default=0.0, help="Dropout between recurrent layers")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Classifier hidden width")
    parser.add_argument("--dropout", type=float, default=0.2, help="Classifier dropout")
    rates = parser.add_mutually_exclusive_group()
    rates.add_argument("--learning-rate", type=float, default=0.001)
    rates.add_argument("--learning-rates", type=float, nargs="+", help="Sequential validation-based comparison")
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", default="64", help="Complete examples per batch, or auto")
    parser.add_argument("--memory-margin", type=float, default=0.2)
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    clipping = parser.add_mutually_exclusive_group()
    clipping.add_argument("--gradient-clip-norm", type=float, default=1.0)
    clipping.add_argument(
        "--no-gradient-clipping", action="store_const", const=None, dest="gradient_clip_norm"
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--evaluate-test", action="store_true", help="Final test evaluation of the selected checkpoint")
    parser.add_argument(
        "--wandb-name", default=os.environ.get("WANDB_NAME"),
        help="Run name/template: {rnn_type}, {rnn_hidden_size}, {learning_rate}, "
             "{rnn_dropout}, {dropout}, {weight_decay}, {seed}, {direction}, {rnn_bidirectional}",
    )
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_RUN_GROUP", DEFAULT_GROUP))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY))
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--wandb-log-model", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    try:
        tracking_config = TrackingConfig(
            project=args.wandb_project, entity=args.wandb_entity,
            mode=args.wandb_mode, log_model=args.wandb_log_model,
        )
        configurations = [RNNTrainingConfig(
            learning_rate=rate, weight_decay=args.weight_decay, dropout=args.dropout,
            hidden_dim=args.hidden_dim, epochs=args.epochs, patience=args.patience,
            min_delta=args.min_delta, seed=args.seed, gradient_clip_norm=args.gradient_clip_norm,
            rnn_type=args.rnn_type, rnn_hidden_size=args.rnn_hidden_size,
            rnn_dropout=args.rnn_dropout, rnn_bidirectional=args.bidirectional,
        ) for rate in (args.learning_rates or [args.learning_rate])]
        run_names = [args.wandb_name.format(
            rnn_type=config.rnn_type, rnn_hidden_size=config.rnn_hidden_size,
            learning_rate=f"{config.learning_rate:g}", seed=config.seed,
            rnn_dropout=f"{config.rnn_dropout:g}", dropout=f"{config.dropout:g}",
            weight_decay=f"{config.weight_decay:g}",
            direction="bi" if config.rnn_bidirectional else "uni",
            rnn_bidirectional=config.rnn_bidirectional,
        ) if args.wandb_name else None for config in configurations]
        batch_size = None if args.batch_size == "auto" else int(args.batch_size)
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch size must be positive")
    except (ValueError, KeyError, IndexError) as error:
        parser.error(str(error))
    if not 0 <= args.memory_margin < 1:
        parser.error("--memory-margin must be between 0 (inclusive) and 1")
    if args.max_batch_size is not None and args.max_batch_size < 1:
        parser.error("--max-batch-size must be positive")
    source_path = (args.inputs_path or args.fasttext_model).expanduser()
    if not source_path.is_file():
        parser.error(f"Input source not found: {source_path}")
    device = select_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is unavailable")

    embedding = vocabulary = None
    if args.inputs_path is not None:
        embedding, vocabulary = load_inputs(source_path)
    data = load_experiment_data(
        cache_dir=args.cache_dir, include_test=args.evaluate_test, vocabulary=vocabulary
    )
    if embedding is None:
        embedding = load_fasttext_embeddings(data.vocabulary, source_path)
    embedding = embedding.to(device)
    splits = {name: getattr(data, name) for name in ("train", "validation")}
    if data.test is not None:
        splits["test"] = data.test
    print(f"Vocabulary size: {len(data.vocabulary)}; fields: {', '.join(TEXT_FIELDS)}")
    print(f"Split sizes: { {name: len(split) for name, split in splits.items()} }")

    if batch_size is None:
        def probe_step(batch):
            # Disposable complete models prevent memory probing from training a trial.
            model = build_rnn_model(embedding, configurations[0]).to(device).train()
            optimizer = build_optimizer(model, configurations[0])
            try:
                loss = torch.nn.functional.cross_entropy(model(batch), batch.labels)
                loss.backward()
                if args.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.gradient_clip_norm, error_if_nonfinite=True
                    )
                optimizer.step()
                return model, optimizer, loss.detach()
            finally:
                optimizer.zero_grad(set_to_none=True)

        # Test inputs never influence capacity selection during tuning.
        longest = max((data.train, data.validation), key=lambda split: split.token_ids.shape[1])
        batch_size = find_gpu_batch_size(
            longest, device, memory_margin=args.memory_margin,
            max_batch_size=min(args.max_batch_size or len(data.train), len(data.train)),
            probe_step=probe_step,
        )
    recurrent_name = f"{'bi' if args.bidirectional else ''}{args.rnn_type}"
    print(f"Device: {device}; batch size: {batch_size}; recurrent model: {recurrent_name}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    experiment_dir = args.output_dir.expanduser() / f"rnn-{stamp}-{uuid4().hex[:8]}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    inputs_path = experiment_dir / "inputs.pt"
    save_inputs(embedding, data.vocabulary, inputs_path)
    versions = reproducibility_metadata(
        embedding, data.vocabulary, entry_point="train_rnn.py",
        model_version=f"frozen-fasttext-{recurrent_name}-mlp-v1",
    )
    group = args.wandb_group or DEFAULT_GROUP
    print(f"Experiment: {experiment_dir}; W&B {tracking_config.mode}; group: {group}", flush=True)
    results = []
    for index, config in enumerate(configurations, start=1):
        seed_everything(config.seed)
        model = build_rnn_model(embedding, config).to(device)
        result = train_classifier(
            None, model,
            make_dataloader(data.train, batch_size, shuffle=True, seed=config.seed),
            make_dataloader(data.validation, batch_size, seed=config.seed),
            device, config,
            experiment_dir / f"trial-{index}-{recurrent_name}-lr-{config.learning_rate:g}",
            inputs_path,
            metadata={
                "dataset": DATASET_NAME,
                "fasttext_model": str(source_path) if args.fasttext_model else None,
                "frozen_inputs_source": str(source_path) if args.inputs_path else None,
                "torch_version": str(torch.__version__), "vocabulary_size": len(data.vocabulary),
                "split_sizes": {name: len(split) for name, split in splits.items()},
                "dataset_split_versions": data.split_versions,
                "device": str(device), "memory_margin": args.memory_margin,
                "batch_size_automatic": args.batch_size == "auto",
                **versions,
            },
            tracking_config=tracking_config, tracking_group=group,
            tracking_name=run_names[index - 1] or (
                f"RNN-{recurrent_name}-h{config.rnn_hidden_size}-lr{config.learning_rate:g}"
                f"-seed{config.seed}-{experiment_dir.name}"
            ),
        )
        results.append(result)
        del model

    selected = min(results, key=lambda result: result["best_validation_loss"])
    summary = {"trials": results, "selected": selected}
    if args.evaluate_test:
        del embedding
        model, _ = load_trained_rnn(selected["best_checkpoint"], device)
        with start_run(
            {"selected": selected, "dataset": DATASET_NAME, "phase": "test",
             "wandb": asdict(tracking_config)},
            f"selected-{recurrent_name}-test", experiment_dir / "evaluation", tracking_config,
            group=group, job_type="evaluate",
        ) as tracking:
            summary["test"] = run_epoch(None, model, make_dataloader(data.test, batch_size), device)
            tracking.log({f"test/{key}": value for key, value in summary["test"].items()})
            tracking.summary.update({"selected": selected})
            summary["evaluation_tracking"] = run_details(tracking, tracking_config.mode)
        print(f"Selected model test metrics: {summary['test']}")
    (experiment_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Best checkpoint: {selected['best_checkpoint']}")


if __name__ == "__main__":
    main()
