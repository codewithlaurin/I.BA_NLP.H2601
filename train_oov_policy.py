"""Train matched UNK/FastText-policy runs with independent validation selection."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from src.art_reasoning.batching import make_dataloader, select_device
from src.art_reasoning.data import (
    DATASET_NAME, _encode_split, _tokenize_split, load_art_dataset, prepare_experiment_data,
)
from src.art_reasoning.embeddings import extend_fasttext_embeddings
from src.art_reasoning.models.classifier import Classifier
from src.art_reasoning.models.classifier_input import ClassifierInput
from src.art_reasoning.preprocessing import FastTextPreprocessor
from src.art_reasoning.tracking import DEFAULT_ENTITY, DEFAULT_PROJECT, TrackingConfig
from src.art_reasoning.training import (
    RNNTrainingConfig, TrainingConfig, build_rnn_model, load_inputs,
    reproducibility_metadata, save_inputs, seed_everything, train_classifier,
)
import torch


def experiment_config(architecture: str, seed: int, epochs: int,
                      baseline_patience: int, rnn_patience: int):
    """Keep each architecture's selected hyperparameters fixed across policies."""
    common = dict(hidden_dim=256, dropout=0.2, weight_decay=0.0001,
                  seed=seed, epochs=epochs, min_delta=0.0)
    if architecture == "mean_pool":
        return TrainingConfig(learning_rate=0.001, patience=baseline_patience, **common)
    if architecture == "gru":
        return RNNTrainingConfig(
            learning_rate=0.003, patience=rnn_patience, rnn_type="gru",
            rnn_hidden_size=256, rnn_dropout=0.4, rnn_bidirectional=False,
            gradient_clip_norm=1.0, **common,
        )
    raise ValueError(f"Unsupported architecture: {architecture}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", dest="architecture", choices=("mean_pool", "gru"), required=True)
    parser.add_argument("--oov-policy", choices=("unk", "fasttext"), required=True)
    parser.add_argument("--inputs-path", type=Path, required=True,
                        help="Original training-only vocabulary and frozen embeddings")
    parser.add_argument("--fasttext-model", type=Path, required=True,
                        help="Same cc.en.300.bin that supplied the saved embeddings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--baseline-patience", type=int, default=3)
    parser.add_argument("--rnn-patience", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oov-selection"))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-entity", default=DEFAULT_ENTITY)
    parser.add_argument("--wandb-group", default="P1-OOV-Selection")
    parser.add_argument("--wandb-log-model", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    try:
        config = experiment_config(args.architecture, args.seed, args.epochs,
                                   args.baseline_patience, args.rnn_patience)
        tracking = TrackingConfig(project=args.wandb_project, entity=args.wandb_entity,
                                  mode=args.wandb_mode, log_model=args.wandb_log_model)
    except ValueError as error:
        parser.error(str(error))
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    source = args.inputs_path.expanduser().resolve()
    fasttext_path = args.fasttext_model.expanduser().resolve()
    for path in (source, fasttext_path):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    device = select_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    embedding, vocabulary = load_inputs(source)
    # No test split is requested, even in the FastText treatment.
    dataset = load_art_dataset(cache_dir=args.cache_dir, include_test=False)
    data = prepare_experiment_data(dataset, include_test=False, vocabulary=vocabulary)
    if (data.train.token_ids == vocabulary.UNK_ID).any().item():
        raise ValueError("Saved vocabulary does not cover all training words; use the original inputs.pt")
    tokens, labels = _tokenize_split(dataset["validation"], FastTextPreprocessor())
    original_vocabulary_size = len(vocabulary)
    original_oov_tokens = int((data.validation.token_ids == vocabulary.UNK_ID).sum().item())
    if args.oov_policy == "fasttext":
        embedding, vocabulary = extend_fasttext_embeddings(
            embedding, vocabulary, tokens, fasttext_path
        )
        data.validation = _encode_split(tokens, labels, vocabulary)
    # Appending rows does not change any training ID or embedding vector.
    # Both policies therefore have identical training inputs before selection.
    embedding = embedding.to(device)
    seed_everything(config.seed)
    if args.architecture == "gru":
        model = build_rnn_model(embedding, config).to(device)
        input_builder = None
    else:
        input_builder = ClassifierInput(embedding).to(device)
        model = Classifier(hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    experiment = args.output_dir.expanduser() / (
        f"{args.architecture}-{args.oov_policy}-seed{args.seed}-{stamp}-{uuid4().hex[:8]}"
    )
    experiment.mkdir(parents=True, exist_ok=False)
    inputs_path = experiment / "inputs.pt"
    # The FastText policy saves its extended cache, so its selected validation
    # checkpoint restores with exactly the vectors used for model selection.
    save_inputs(embedding, vocabulary, inputs_path)
    metadata = {
        "dataset": DATASET_NAME, "model_family": args.architecture, "oov_policy": args.oov_policy,
        "phase": "train_validation", "selection_metric": "validation/loss",
        "frozen_inputs_source": str(source), "fasttext_model": str(fasttext_path),
        "device": str(device), "split_sizes": {"train": len(data.train), "validation": len(data.validation)},
        "dataset_split_versions": data.split_versions,
        "original_vocabulary_size": original_vocabulary_size, "vocabulary_size": len(vocabulary),
        "original_validation_oov_tokens": original_oov_tokens,
        "validation_oov_tokens": int((data.validation.token_ids == vocabulary.UNK_ID).sum().item()),
        **reproducibility_metadata(embedding, vocabulary, entry_point="train_oov_policy.py",
                                   model_version=f"frozen-fasttext-{args.architecture}-oov-policy-v1"),
    }
    print(f"Experiment: {experiment}; policy={args.oov_policy}; device={device}; "
          f"validation UNK tokens={metadata['validation_oov_tokens']}", flush=True)
    result = train_classifier(
        input_builder, model,
        make_dataloader(data.train, args.batch_size, shuffle=True, seed=config.seed),
        make_dataloader(data.validation, args.batch_size, seed=config.seed),
        device, config, experiment / "trial", inputs_path, metadata=metadata,
        tracking_config=tracking, tracking_group=args.wandb_group,
        tracking_name=f"OOV-{args.architecture}-{args.oov_policy}-seed{args.seed}",
    )
    (experiment / "summary.json").write_text(json.dumps({
        "architecture": args.architecture, "oov_policy": args.oov_policy,
        "seed": args.seed, "selected": result,
    }, indent=2) + "\n")
    print(f"Selected checkpoint: {result['best_checkpoint']}")


if __name__ == "__main__":
    main()
