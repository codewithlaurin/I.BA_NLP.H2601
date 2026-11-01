"""Shared training and checkpointing for mean-pooled and recurrent ART models."""

from dataclasses import asdict, dataclass, fields
import json
import hashlib
from importlib.metadata import version
import math
import os
from pathlib import Path
import random
import platform
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import TEXT_FIELDS
from .embeddings import EMBEDDING_DIM
from .models.classifier import Classifier
from .models.classifier_input import ClassifierInput
from .models.rnn_classifier import RNNClassifier
from .models.sequence_encoder import SequenceEncoder
from .tracking import TrackingConfig, log_best_model, run_details, start_run
from .vocabulary import Vocabulary


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    hidden_dim: int = 128
    epochs: int = 30
    patience: int = 5
    min_delta: float = 0.0
    seed: int = 42
    gradient_clip_norm: float | None = None
    save_accuracy_checkpoint: bool = False

    def __post_init__(self):
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 (inclusive) and 1")
        if min(self.hidden_dim, self.epochs, self.patience) < 1:
            raise ValueError("hidden_dim, epochs, and patience must be positive")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1")
        if self.gradient_clip_norm is not None and (
            not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive, or None")


@dataclass(frozen=True)
class RNNTrainingConfig(TrainingConfig):
    hidden_dim: int = 256
    gradient_clip_norm: float | None = 1.0
    rnn_type: str = "gru"
    rnn_hidden_size: int = 128
    rnn_dropout: float = 0.0
    rnn_bidirectional: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.rnn_type not in ("gru", "lstm"):
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        if not isinstance(self.rnn_hidden_size, int) or isinstance(self.rnn_hidden_size, bool) or self.rnn_hidden_size < 1:
            raise ValueError("rnn_hidden_size must be a positive integer")
        if not 0 <= self.rnn_dropout < 1:
            raise ValueError("rnn_dropout must be between 0 (inclusive) and 1")
        if not isinstance(self.rnn_bidirectional, bool):
            raise ValueError("rnn_bidirectional must be a boolean")


def build_rnn_model(embedding: nn.Embedding, config: RNNTrainingConfig) -> RNNClassifier:
    """Initialize fresh recurrent and head weights around the frozen embedding."""
    return RNNClassifier(
        SequenceEncoder(
            embedding, hidden_size=config.rnn_hidden_size,
            rnn_type=config.rnn_type, dropout=config.rnn_dropout,
            bidirectional=config.rnn_bidirectional,
        ),
        hidden_dim=config.hidden_dim, dropout=config.dropout,
    )


def seed_everything(seed: int) -> None:
    """Reuse initialization and shuffle seeds across learning-rate trials."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def reproducibility_metadata(
    input_builder: ClassifierInput | SequenceEncoder | nn.Embedding,
    vocabulary: Vocabulary,
    *,
    entry_point: str = "train_classifier.py",
    model_version: str = "frozen-fasttext-mean-pool-mlp-v1",
) -> dict:
    """Identify the actual input vectors, ordered vocabulary, code, and libraries.

    Hash the compact task embedding matrix instead of rereading the entire
    multi-gigabyte pretrained binary for every learning-rate trial.
    """
    root = Path(__file__).resolve().parents[2]
    code_hash = hashlib.sha256()
    paths = sorted((root / "src" / "art_reasoning").rglob("*.py"))
    for path in [root / entry_point, *paths]:
        code_hash.update(str(path.relative_to(root)).encode())
        code_hash.update(path.read_bytes())
    embedding = input_builder if isinstance(input_builder, nn.Embedding) else input_builder.embedding
    weights = embedding.weight.detach().cpu().contiguous().numpy()
    return {
        "model_version": model_version,
        "entry_point": entry_point,
        "model_code_sha256": code_hash.hexdigest(),
        "embedding_weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "vocabulary_sha256": hashlib.sha256(
            json.dumps(list(vocabulary.id_to_token), ensure_ascii=False).encode()
        ).hexdigest(),
        "python_version": platform.python_version(),
        "library_versions": {
            package: version(package)
            for package in ("torch", "datasets", "numpy", "fasttext", "wandb", "huggingface-hub")
        },
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def build_optimizer(classifier: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """Decay trainable weights; exclude dense/recurrent biases and frozen embeddings."""
    weights, biases = [], []
    for name, parameter in classifier.named_parameters():
        if parameter.requires_grad:
            local_name = name.rsplit(".", 1)[-1]
            is_bias = local_name == "bias" or local_name.startswith("bias_")
            (biases if is_bias else weights).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": weights, "weight_decay": config.weight_decay},
            {"params": biases, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )


def run_epoch(
    input_builder: ClassifierInput | None,
    classifier: Classifier | RNNClassifier,
    loader: DataLoader,
    device: torch.device | str,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
) -> dict:
    """Return example-weighted loss, accuracy, and count; update only in training.

    Logits have shape (B, 2); integer labels have shape (B,) with values 0/1.
    CrossEntropyLoss receives raw logits, with no preceding softmax or sigmoid.
    Pass input_builder=None for an RNNClassifier that owns its encoder.
    """
    training = optimizer is not None
    if input_builder is not None:
        input_builder.eval()
    classifier.train(training)
    criterion = nn.CrossEntropyLoss()
    loss_sum = correct = examples = 0
    gradient_norm_sum = gradient_norm_max = 0.0
    gradient_steps = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = classifier(batch) if input_builder is None else classifier(input_builder(batch))
            loss = criterion(logits, batch.labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss; check inputs and learning rate")
            if training:
                loss.backward()
                if gradient_clip_norm is not None:
                    norm = nn.utils.clip_grad_norm_(
                        classifier.parameters(), gradient_clip_norm, error_if_nonfinite=True
                    ).item()
                    gradient_norm_sum += norm
                    gradient_norm_max = max(gradient_norm_max, norm)
                    gradient_steps += 1
                optimizer.step()
            count = batch.labels.numel()
            loss_sum += loss.detach().item() * count
            correct += (logits.detach().argmax(dim=-1) == batch.labels).sum().item()
            examples += count
    if not examples:
        raise ValueError("Cannot run an epoch on an empty split")
    metrics = {"loss": loss_sum / examples, "accuracy": correct / examples, "examples": examples}
    if gradient_steps:
        metrics.update(
            gradient_norm_mean=gradient_norm_sum / gradient_steps,
            gradient_norm_max=gradient_norm_max,
        )
    return metrics


def _save_checkpoint(payload: dict, path: Path) -> None:
    """Replace a completed checkpoint atomically, retaining the previous on failure."""
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_inputs(
    input_builder: ClassifierInput | SequenceEncoder | nn.Embedding,
    vocabulary: Vocabulary,
    path: Path,
) -> None:
    """Store frozen embeddings once per experiment, shared by all trial checkpoints."""
    embedding = input_builder if isinstance(input_builder, nn.Embedding) else input_builder.embedding
    _save_checkpoint(
        {
            "embedding_weight": embedding.weight.detach().cpu(),
            "id_to_token": list(vocabulary.id_to_token),
            "text_fields": list(TEXT_FIELDS),
            "label_to_class": {1: 0, 2: 1},
            "preprocessing": "FastTextPreprocessor: lowercase, remove punctuation, split whitespace",
        },
        path,
    )


def train_classifier(
    input_builder: ClassifierInput | None,
    classifier: Classifier | RNNClassifier,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device | str,
    config: TrainingConfig,
    run_dir: Path,
    inputs_path: Path,
    metadata: dict | None = None,
    tracking_config: TrackingConfig | None = None,
    tracking_group: str | None = None,
    tracking_name: str | None = None,
) -> dict:
    """Select the lowest validation loss, checkpoint each epoch, and stop on a plateau.

    The caller seeds and creates a fresh model and loaders for each trial.
    The historical function name is retained for baseline compatibility;
    pass input_builder=None and a complete RNNClassifier for recurrent training.
    The test split is deliberately absent from this function.
    """
    tracking_config = tracking_config or TrackingConfig()
    recurrent = isinstance(classifier, RNNClassifier)
    if recurrent and (input_builder is not None or not isinstance(config, RNNTrainingConfig)):
        raise ValueError("RNN training requires input_builder=None and RNNTrainingConfig")
    if not recurrent and input_builder is None:
        raise ValueError("Mean-pooling training requires its ClassifierInput")
    encoder = classifier.encoder if recurrent else input_builder
    run_dir.mkdir(parents=True, exist_ok=False)
    optimizer = build_optimizer(classifier, config)
    settings = {
        **(metadata or {}), **asdict(config), "optimizer": "AdamW",
        "loss": "CrossEntropyLoss", "wandb": asdict(tracking_config),
        "wandb_group": tracking_group,
        "wandb_name": tracking_name or run_dir.name,
        "batch_size": train_loader.batch_size,
        "optimizer_settings": {
            "betas": list(optimizer.defaults["betas"]), "eps": optimizer.defaults["eps"],
            "amsgrad": optimizer.defaults["amsgrad"], "bias_weight_decay": 0.0,
        },
        "model_type": "rnn" if recurrent else "mean_pool",
        "model_family": "rnn" if recurrent else "mean_pool",
        "architecture": str(classifier),
        "pooling": "top_layer_final_hidden_then_concatenate" if recurrent else "masked_mean_then_concatenate",
        "embedding_dim": encoder.embedding.embedding_dim, "embeddings_frozen": True,
        "trainable_parameters": sum(p.numel() for p in classifier.parameters() if p.requires_grad),
        "frozen_embedding_parameters": encoder.embedding.weight.numel(),
        "text_fields": list(TEXT_FIELDS), "output_dim": 2,
        "label_smoothing": 0.0, "scheduler": None,
    }
    if recurrent:
        settings.update(
            rnn_num_layers=classifier.encoder.rnn.num_layers,
            rnn_bidirectional=classifier.encoder.rnn.bidirectional,
            rnn_num_directions=classifier.encoder.num_directions,
            classifier_input_dim=classifier.encoder.output_dim,
        )
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    best_loss = stopping_best = math.inf
    best_accuracy = None
    best_epoch = bad_epochs = 0
    peak_accuracy, peak_accuracy_loss, peak_accuracy_epoch = -math.inf, math.inf, 0
    with start_run(
        settings, tracking_name or run_dir.name, run_dir, tracking_config, tracking_group
    ) as tracking, (run_dir / "metrics.jsonl").open("w") as metrics_file:
        identity = run_details(tracking, tracking_config.mode)
        (run_dir / "wandb.json").write_text(json.dumps(identity, indent=2) + "\n")
        if identity["wandb_run_url"]:
            print(f"W&B run: {identity['wandb_run_url']}", flush=True)
        # W&B initialization must not change the seed used by the training loop.
        seed_everything(config.seed)
        training_started = perf_counter()
        for epoch in range(1, config.epochs + 1):
            train_metrics = run_epoch(
                input_builder, classifier, train_loader, device, optimizer,
                gradient_clip_norm=config.gradient_clip_norm,
            )
            validation_metrics = run_epoch(input_builder, classifier, validation_loader, device)
            metrics = {
                "epoch": epoch, "learning_rate": config.learning_rate,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"validation/{key}": value for key, value in validation_metrics.items()},
            }
            current_loss = validation_metrics["loss"]
            improved = current_loss < best_loss
            if improved:
                best_loss, best_epoch = current_loss, epoch
                best_accuracy = validation_metrics["accuracy"]
            # Keep accuracy selection independent of loss selection and patience.
            current_accuracy = validation_metrics["accuracy"]
            accuracy_improved = (current_accuracy > peak_accuracy or
                                 (current_accuracy == peak_accuracy and current_loss < peak_accuracy_loss))
            if accuracy_improved:
                peak_accuracy, peak_accuracy_loss, peak_accuracy_epoch = current_accuracy, current_loss, epoch
            # min_delta affects patience only: always retain the true best model.
            if current_loss < stopping_best - config.min_delta:
                stopping_best, bad_epochs = current_loss, 0
            else:
                bad_epochs += 1
            checkpoint = {
                "checkpoint_format_version": 2 if recurrent else 1,
                "model_type": settings["model_type"],
                "optimizer_state": optimizer.state_dict(),
                "config": settings, "epoch": epoch, "metrics": metrics,
                "best_validation_loss": best_loss, "best_epoch": best_epoch,
                "best_validation_accuracy": best_accuracy,
                "bad_epochs": bad_epochs,
                "inputs_file": os.path.relpath(inputs_path, run_dir),
            }
            if config.save_accuracy_checkpoint:
                checkpoint.update(
                    peak_validation_accuracy=peak_accuracy,
                    peak_accuracy_epoch=peak_accuracy_epoch,
                    validation_loss_at_peak_accuracy=peak_accuracy_loss,
                )
            if recurrent:
                # Frozen inputs are stored once, not duplicated in each checkpoint.
                checkpoint["model_state"] = {
                    name: value for name, value in classifier.state_dict().items()
                    if name != "encoder.embedding.weight"
                }
            else:
                checkpoint["classifier_state"] = classifier.state_dict()
            _save_checkpoint(checkpoint, run_dir / "last.pt")
            if improved:
                _save_checkpoint(checkpoint, run_dir / "best.pt")
            if config.save_accuracy_checkpoint and accuracy_improved:
                _save_checkpoint(
                    {**checkpoint, "checkpoint_selection_metric": "validation_accuracy"},
                    run_dir / "best_accuracy.pt",
                )
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
            tracking.log(metrics, step=epoch)
            print(
                f"{run_dir.name} epoch {epoch}: train loss={train_metrics['loss']:.4f}, "
                f"validation loss={current_loss:.4f}, accuracy={validation_metrics['accuracy']:.3f}",
                flush=True,
            )
            if bad_epochs >= config.patience:
                print(f"Early stopping after {bad_epochs} validation checks without sufficient improvement.")
                break
        result = {
            "best_validation_loss": best_loss, "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "epochs_completed": epoch, "stopped_early": epoch < config.epochs,
            "best_checkpoint": str(run_dir / "best.pt"),
            "learning_rate": config.learning_rate,
            "training_duration_seconds": perf_counter() - training_started,
            **identity,
        }
        if config.save_accuracy_checkpoint:
            result.update(
                peak_validation_accuracy=peak_accuracy,
                peak_accuracy_epoch=peak_accuracy_epoch,
                validation_loss_at_peak_accuracy=peak_accuracy_loss,
                best_accuracy_checkpoint=str(run_dir / "best_accuracy.pt"),
            )
        if tracking_config.log_model and tracking_config.mode != "disabled":
            log_best_model(
                tracking, run_dir / "best.pt", inputs_path, model_type=settings["model_type"],
                accuracy_checkpoint=run_dir / "best_accuracy.pt" if config.save_accuracy_checkpoint else None,
            )
        tracking.summary.update(result)
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_inputs(path: Path | str) -> tuple[nn.Embedding, Vocabulary]:
    """Restore a saved frozen embedding and its exact training-vocabulary mapping."""
    inputs = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    vocabulary = Vocabulary([inputs["id_to_token"][2:]])
    if list(vocabulary.id_to_token) != inputs["id_to_token"] or inputs["text_fields"] != list(TEXT_FIELDS):
        raise ValueError("Saved vocabulary or text-field order is incompatible")
    if inputs["label_to_class"] != {1: 0, 2: 1}:
        raise ValueError("Saved label mapping is incompatible")
    if inputs["embedding_weight"].shape != (len(vocabulary), EMBEDDING_DIM):
        raise ValueError("Saved embedding shape does not match the vocabulary and dimension")
    embedding = nn.Embedding.from_pretrained(
        inputs["embedding_weight"], freeze=True, padding_idx=Vocabulary.PAD_ID
    )
    return embedding, vocabulary


def load_trained_classifier(
    checkpoint_path: Path | str, device: torch.device | str = "cpu"
) -> tuple[ClassifierInput, Classifier, Vocabulary]:
    """Restore current or historical mean-pooling checkpoints for inference."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type", "mean_pool") != "mean_pool":
        raise ValueError("Use load_trained_rnn() for a recurrent checkpoint")
    embedding, vocabulary = load_inputs(checkpoint_path.parent / checkpoint["inputs_file"])
    input_builder = ClassifierInput(embedding).to(device).eval()
    settings = checkpoint["config"]
    classifier = Classifier(hidden_dim=settings["hidden_dim"], dropout=settings["dropout"])
    classifier.load_state_dict(checkpoint["classifier_state"])
    return input_builder, classifier.to(device).eval(), vocabulary


def load_trained_rnn(
    checkpoint_path: Path | str, device: torch.device | str = "cpu"
) -> tuple[RNNClassifier, Vocabulary]:
    """Restore GRU/LSTM and head using only the checkpoint and shared inputs.pt."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "rnn" or checkpoint.get("checkpoint_format_version") != 2:
        raise ValueError("Expected a version-2 recurrent checkpoint")
    embedding, vocabulary = load_inputs(checkpoint_path.parent / checkpoint["inputs_file"])
    names = {field.name for field in fields(RNNTrainingConfig)}
    # Historical checkpoints omit the direction or record False in metadata.
    # The dataclass default reconstructs their original forward-only architecture.
    config = RNNTrainingConfig(**{key: value for key, value in checkpoint["config"].items() if key in names})
    model = build_rnn_model(embedding, config)
    model.load_state_dict({**checkpoint["model_state"], "encoder.embedding.weight": embedding.weight})
    return model.to(device).eval(), vocabulary
