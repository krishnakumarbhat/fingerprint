from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class RuntimeConfig:
    image_width: int = 96
    image_height: int = 103
    embedding_dim: int = 128
    batch_size: int = 48
    eval_batch_size: int = 96
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = torch.cuda.is_available()


@dataclass(frozen=True)
class TrainingConfig:
    baseline_epochs: int = 2
    search_epochs: int = 2
    final_epochs: int = 2
    samples_per_identity: int = 4
    train_steps_per_epoch: int = 120
    search_steps_per_epoch: int = 60
    learning_rates: tuple[float, ...] = (1e-3, 3e-4)
    batch_sizes: tuple[int, ...] = (32, 48)
    margins: tuple[float, ...] = (0.75, 1.0)
    weight_decay: float = 1e-4
    early_stopping_patience: int = 2
    max_impostor_pairs_per_identity: int = 128


@dataclass(frozen=True)
class ExperimentGrid:
    preprocess_variants: tuple[str, ...] = (
        "raw",
        "segmented",
        "normalized",
        "gabor",
        "skeleton",
    )
    augmentation_variants: tuple[str, ...] = (
        "none",
        "rotate",
        "elastic",
        "obliterate",
        "combined",
    )
    model_variants: tuple[str, ...] = (
        "compact_siamese",
        "mobile_triplet",
    )


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    dataset_root: Path
    output_root: Path
    split: SplitConfig = field(default_factory=SplitConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grid: ExperimentGrid = field(default_factory=ExperimentGrid)

    @classmethod
    def from_root(cls, project_root: Path) -> "AppConfig":
        return cls(
            project_root=project_root,
            dataset_root=project_root / "archive" / "SOCOFing",
            output_root=project_root / "artifacts",
        )