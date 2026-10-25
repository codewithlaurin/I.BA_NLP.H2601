"""Train and validate the frozen-FastText classifier with plain PyTorch."""

from dataclasses import asdict, dataclass
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
from .models.classifier import Classifier
from .models.classifier_input import ClassifierInput
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


def seed_everything(seed: int) -> None:
    """Reuse initialization and shuffle seeds across learning-rate trials."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def reproducibility_metadata(
    input_builder: ClassifierInput, vocabulary: Vocabulary
) -> dict:
    """Identify the actual input vectors, ordered vocabulary, code, and libraries.

    Hash the compact task embedding matrix instead of rereading the entire
    multi-gigabyte pretrained binary for every learning-rate trial.
    """
    root = Path(__file__).resolve().parents[2]
    code_hash = hashlib.sha256()
    paths = sorted((root / "src" / "art_reasoning").rglob("*.py"))
    for path in [root / "train_classifier.py", *paths]:
        code_hash.update(str(path.relative_to(root)).encode())
        code_hash.update(path.read_bytes())
    weights = input_builder.embedding.weight.detach().cpu().contiguous().numpy()
    return {
        "model_version": "frozen-fasttext-mean-pool-mlp-v1",
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


def build_optimizer(classifier: Classifier, config: TrainingConfig) -> torch.optim.AdamW:
    """Decay dense weights only; exclude biases and frozen parameters."""
    weights, biases = [], []
    for name, parameter in classifier.named_parameters():
        if parameter.requires_grad:
            (biases if name.endswith("bias") else weights).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": weights, "weight_decay": config.weight_decay},
            {"params": biases, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )


def run_epoch(
    input_builder: ClassifierInput,
    classifier: Classifier,
    loader: DataLoader,
    device: torch.device | str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Return example-weighted loss, accuracy, and count; update only in training.

    Logits have shape (B, 2); integer labels have shape (B,) with values 0/1.
    CrossEntropyLoss receives raw logits, with no preceding softmax or sigmoid.
    """
    training = optimizer is not None
    input_builder.eval()
    classifier.train(training)
    criterion = nn.CrossEntropyLoss()
    loss_sum = correct = examples = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = classifier(input_builder(batch))
            loss = criterion(logits, batch.labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss; check inputs and learning rate")
            if training:
                loss.backward()
                optimizer.step()
            count = batch.labels.numel()
            loss_sum += loss.detach().item() * count
            correct += (logits.detach().argmax(dim=-1) == batch.labels).sum().item()
            examples += count
    if not examples:
        raise ValueError("Cannot run an epoch on an empty split")
    return {"loss": loss_sum / examples, "accuracy": correct / examples, "examples": examples}


def _save_checkpoint(payload: dict, path: Path) -> None:
    """Replace a completed checkpoint atomically, retaining the previous on failure."""
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_inputs(input_builder: ClassifierInput, vocabulary: Vocabulary, path: Path) -> None:
    """Store frozen embeddings once per experiment, shared by all trial checkpoints."""
    _save_checkpoint(
        {
            "embedding_weight": input_builder.embedding.weight.detach().cpu(),
            "id_to_token": list(vocabulary.id_to_token),
            "text_fields": list(TEXT_FIELDS),
            "label_to_class": {1: 0, 2: 1},
            "preprocessing": "FastTextPreprocessor: lowercase, remove punctuation, split whitespace",
        },
        path,
    )


def train_classifier(
    input_builder: ClassifierInput,
    classifier: Classifier,
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

    The caller seeds and creates a fresh classifier and loaders for each trial.
    The test split is deliberately absent from this function.
    """
    tracking_config = tracking_config or TrackingConfig()
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
        "architecture": str(classifier), "pooling": "masked_mean_then_concatenate",
        "embedding_dim": input_builder.embedding.embedding_dim, "embeddings_frozen": True,
        "text_fields": list(TEXT_FIELDS), "output_dim": 2,
        "label_smoothing": 0.0, "scheduler": None,
    }
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    best_loss = stopping_best = math.inf
    best_accuracy = None
    best_epoch = bad_epochs = 0
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
            train_metrics = run_epoch(input_builder, classifier, train_loader, device, optimizer)
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
            # min_delta affects patience only: always retain the true best model.
            if current_loss < stopping_best - config.min_delta:
                stopping_best, bad_epochs = current_loss, 0
            else:
                bad_epochs += 1
            checkpoint = {
                "classifier_state": classifier.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": settings, "epoch": epoch, "metrics": metrics,
                "best_validation_loss": best_loss, "best_epoch": best_epoch,
                "best_validation_accuracy": best_accuracy,
                "bad_epochs": bad_epochs,
                "inputs_file": os.path.relpath(inputs_path, run_dir),
            }
            _save_checkpoint(checkpoint, run_dir / "last.pt")
            if improved:
                _save_checkpoint(checkpoint, run_dir / "best.pt")
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
        if tracking_config.log_model and tracking_config.mode != "disabled":
            log_best_model(tracking, run_dir / "best.pt", inputs_path)
        tracking.summary.update(result)
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_trained_classifier(
    checkpoint_path: Path | str, device: torch.device | str = "cpu"
) -> tuple[ClassifierInput, Classifier, Vocabulary]:
    """Restore inference models and vocabulary; keep inputs.pt beside trial folders."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    inputs = torch.load(
        checkpoint_path.parent / checkpoint["inputs_file"], map_location="cpu", weights_only=True
    )
    vocabulary = Vocabulary([inputs["id_to_token"][2:]])
    if list(vocabulary.id_to_token) != inputs["id_to_token"] or inputs["text_fields"] != list(TEXT_FIELDS):
        raise ValueError("Checkpoint vocabulary or text-field order is incompatible")
    embedding = nn.Embedding.from_pretrained(
        inputs["embedding_weight"], freeze=True, padding_idx=Vocabulary.PAD_ID
    )
    input_builder = ClassifierInput(embedding).to(device).eval()
    settings = checkpoint["config"]
    classifier = Classifier(hidden_dim=settings["hidden_dim"], dropout=settings["dropout"])
    classifier.load_state_dict(checkpoint["classifier_state"])
    return input_builder, classifier.to(device).eval(), vocabulary
