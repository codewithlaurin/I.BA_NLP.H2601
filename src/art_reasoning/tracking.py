"""Weights & Biases runs, metric definitions, and model artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import wandb


DEFAULT_PROJECT = "nlp-classifier"
DEFAULT_ENTITY = "laurin-uni-workspace"


@dataclass(frozen=True)
class TrackingConfig:
    project: str = DEFAULT_PROJECT
    entity: str | None = DEFAULT_ENTITY
    mode: Literal["online", "offline", "disabled"] = "online"
    log_model: bool = True

    def __post_init__(self):
        if not self.project.strip():
            raise ValueError("W&B project must not be empty")
        if self.entity is not None and not self.entity.strip():
            raise ValueError("W&B entity must not be empty")
        if self.mode not in ("online", "offline", "disabled"):
            raise ValueError("W&B mode must be online, offline, or disabled")


def start_run(
    config: dict,
    name: str,
    run_dir: Path,
    tracking: TrackingConfig,
    group: str | None = None,
    job_type: str = "train",
) -> wandb.Run:
    """Start a run; credentials come from wandb login or WANDB_API_KEY.

    Use the returned run as a context manager so success and exceptions both
    finish the run. Never pass credentials in config: that dictionary is logged.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=tracking.project,
        entity=tracking.entity,
        name=name,
        group=group,
        job_type=job_type,
        config=config,
        mode=tracking.mode,
        dir=str(run_dir.resolve()),
    )
    run.define_metric("epoch")
    for name in ("train/*", "validation/*", "learning_rate"):
        run.define_metric(name, step_metric="epoch")
    return run


def run_details(run: wandb.Run, mode: str) -> dict:
    """Keep the run identity with local results; offline runs have no cloud URL yet."""
    return {
        "wandb_run_id": run.id if mode != "disabled" else None,
        "wandb_run_url": run.url if mode == "online" else None,
    }


def log_best_model(run: wandb.Run, checkpoint: Path, inputs_path: Path) -> None:
    """Upload one restorable best-model bundle after training, including frozen inputs."""
    artifact = wandb.Artifact(
        name=f"classifier-{run.id}",
        type="model",
        metadata={"checkpoint_metric": "validation/loss", "checkpoint_mode": "min"},
    )
    # Preserve the checkpoint's ../inputs.pt reference when downloading the bundle.
    artifact.add_file(str(checkpoint), name="trial/best.pt")
    artifact.add_file(str(inputs_path), name="inputs.pt")
    run.log_artifact(artifact, aliases=["best"])
