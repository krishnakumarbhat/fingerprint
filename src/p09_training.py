from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.p01_config import RuntimeConfig, SplitConfig, TrainingConfig
from src.p03_dataset_index import FingerprintRecord
from src.p04_phase_one_split import FingerprintDataset, IdentityBatchSampler, build_label_map
from src.p05_phase_two_preprocessing import apply_preprocess_variant
from src.p06_phase_three_augmentation import apply_augmentation_variant
from src.p07_phase_four_models import create_model
from src.p08_metric_learning import LossSummary, compute_metric_loss
from src.p10_phase_five_evaluation import VerificationMetrics, compute_verification_metrics


@dataclass(frozen=True)
class EpochHistory:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_eer: float
    validation_accuracy: float
    positive_distance: float
    negative_distance: float


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    model_name: str
    preprocess_variant: str
    augmentation_variant: str
    learning_rate: float
    batch_size: int
    margin: float
    objective_name: str
    best_epoch: int
    history: list[EpochHistory]
    validation_metrics: VerificationMetrics
    test_metrics: VerificationMetrics
    checkpoint_path: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["validation_metrics"] = asdict(self.validation_metrics)
        data["test_metrics"] = asdict(self.test_metrics)
        data["history"] = [asdict(item) for item in self.history]
        return data


def _device_type(device_name: str) -> str:
    return "cuda" if device_name.startswith("cuda") else "cpu"


def _create_loader(
    records: list[FingerprintRecord],
    batch_size: int,
    runtime_config: RuntimeConfig,
    split_config: SplitConfig,
    steps_per_epoch: int | None,
    training_config: TrainingConfig,
    balanced: bool,
) -> tuple[DataLoader, FingerprintDataset]:
    label_map = build_label_map(records)
    dataset = FingerprintDataset(
        records,
        label_map,
        target_size=(runtime_config.image_height, runtime_config.image_width),
    )
    if balanced:
        sampler = IdentityBatchSampler(
            records=records,
            batch_size=batch_size,
            samples_per_identity=training_config.samples_per_identity,
            steps_per_epoch=steps_per_epoch or 1,
            seed=split_config.seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=runtime_config.num_workers,
            pin_memory=runtime_config.pin_memory,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=runtime_config.eval_batch_size,
            shuffle=False,
            num_workers=runtime_config.num_workers,
            pin_memory=runtime_config.pin_memory,
        )
    return loader, dataset


def _run_loss_epoch(
    model: nn.Module,
    loader: DataLoader,
    objective_name: str,
    preprocess_variant: str,
    augmentation_variant: str,
    margin: float,
    device_name: str,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[float, float, float]:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()
    loss_total = 0.0
    positive_total = 0.0
    negative_total = 0.0
    batches = 0
    device_type = _device_type(device_name)
    for images, labels, _ in loader:
        images = images.to(device_name, non_blocking=True)
        labels = labels.to(device_name, non_blocking=True)
        if training:
            images = apply_augmentation_variant(images, augmentation_variant)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=device_type == "cuda"):
            processed = apply_preprocess_variant(images, preprocess_variant)
            embeddings = model(processed)
            summary = compute_metric_loss(objective_name, embeddings, labels, margin)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(summary.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        loss_total += float(summary.loss.detach().cpu().item())
        positive_total += summary.positive_distance
        negative_total += summary.negative_distance
        batches += 1
    divisor = max(batches, 1)
    return loss_total / divisor, positive_total / divisor, negative_total / divisor


def collect_embeddings(
    model: nn.Module,
    loader: DataLoader,
    dataset: FingerprintDataset,
    preprocess_variant: str,
    device_name: str,
) -> tuple[np.ndarray, list[FingerprintRecord]]:
    model.eval()
    device_type = _device_type(device_name)
    embeddings: list[torch.Tensor] = []
    records: list[FingerprintRecord] = []
    with torch.no_grad():
        for images, _, indices in loader:
            images = images.to(device_name, non_blocking=True)
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=device_type == "cuda"):
                processed = apply_preprocess_variant(images, preprocess_variant)
                batch_embeddings = model(processed)
            embeddings.append(batch_embeddings.detach().cpu())
            records.extend(dataset.records[int(index)] for index in indices.tolist())
    return torch.cat(embeddings, dim=0).numpy(), records


def train_experiment(
    model_name: str,
    preprocess_variant: str,
    augmentation_variant: str,
    train_records: list[FingerprintRecord],
    validation_records: list[FingerprintRecord],
    test_records: list[FingerprintRecord],
    runtime_config: RuntimeConfig,
    training_config: TrainingConfig,
    split_config: SplitConfig,
    learning_rate: float,
    batch_size: int,
    margin: float,
    epochs: int,
    steps_per_epoch: int,
    checkpoint_dir: Path,
) -> ExperimentResult:
    experiment_id = (
        f"{model_name}__{preprocess_variant}__{augmentation_variant}__lr{learning_rate:.0e}__bs{batch_size}__m{margin:.2f}"
        .replace(".", "p")
        .replace("-", "m")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{experiment_id}.pt"
    model, objective_name = create_model(model_name, runtime_config.embedding_dim)
    model.to(runtime_config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=training_config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime_config.device.startswith("cuda"))
    train_loader, _ = _create_loader(
        records=train_records,
        batch_size=batch_size,
        runtime_config=runtime_config,
        split_config=split_config,
        steps_per_epoch=steps_per_epoch,
        training_config=training_config,
        balanced=True,
    )
    validation_loss_steps = max(1, steps_per_epoch // 2)
    validation_loss_loader, _ = _create_loader(
        records=validation_records,
        batch_size=batch_size,
        runtime_config=runtime_config,
        split_config=split_config,
        steps_per_epoch=validation_loss_steps,
        training_config=training_config,
        balanced=True,
    )
    validation_eval_loader, validation_dataset = _create_loader(
        records=validation_records,
        batch_size=batch_size,
        runtime_config=runtime_config,
        split_config=split_config,
        steps_per_epoch=None,
        training_config=training_config,
        balanced=False,
    )
    test_eval_loader, test_dataset = _create_loader(
        records=test_records,
        batch_size=batch_size,
        runtime_config=runtime_config,
        split_config=split_config,
        steps_per_epoch=None,
        training_config=training_config,
        balanced=False,
    )
    history: list[EpochHistory] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: VerificationMetrics | None = None
    best_epoch = 1
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss, positive_distance, negative_distance = _run_loss_epoch(
            model=model,
            loader=train_loader,
            objective_name=objective_name,
            preprocess_variant=preprocess_variant,
            augmentation_variant=augmentation_variant,
            margin=margin,
            device_name=runtime_config.device,
            optimizer=optimizer,
            scaler=scaler,
        )
        validation_loss, _, _ = _run_loss_epoch(
            model=model,
            loader=validation_loss_loader,
            objective_name=objective_name,
            preprocess_variant=preprocess_variant,
            augmentation_variant="none",
            margin=margin,
            device_name=runtime_config.device,
            optimizer=None,
            scaler=scaler,
        )
        validation_embeddings, validation_eval_records = collect_embeddings(
            model=model,
            loader=validation_eval_loader,
            dataset=validation_dataset,
            preprocess_variant=preprocess_variant,
            device_name=runtime_config.device,
        )
        validation_metrics = compute_verification_metrics(
            embeddings=validation_embeddings,
            records=validation_eval_records,
            max_impostor_pairs_per_identity=training_config.max_impostor_pairs_per_identity,
            seed=split_config.seed,
        )
        history.append(
            EpochHistory(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                validation_eer=validation_metrics.eer,
                validation_accuracy=validation_metrics.accuracy,
                positive_distance=positive_distance,
                negative_distance=negative_distance,
            )
        )
        if best_metrics is None or validation_metrics.eer < best_metrics.eer:
            best_metrics = validation_metrics
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(best_state, checkpoint_path)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.early_stopping_patience:
                break
    if best_state is None or best_metrics is None:
        raise RuntimeError(f"Training failed to produce a checkpoint for {experiment_id}")
    model.load_state_dict(best_state)
    model.to(runtime_config.device)
    test_embeddings, test_eval_records = collect_embeddings(
        model=model,
        loader=test_eval_loader,
        dataset=test_dataset,
        preprocess_variant=preprocess_variant,
        device_name=runtime_config.device,
    )
    test_metrics = compute_verification_metrics(
        embeddings=test_embeddings,
        records=test_eval_records,
        max_impostor_pairs_per_identity=training_config.max_impostor_pairs_per_identity,
        seed=split_config.seed,
    )
    return ExperimentResult(
        experiment_id=experiment_id,
        model_name=model_name,
        preprocess_variant=preprocess_variant,
        augmentation_variant=augmentation_variant,
        learning_rate=learning_rate,
        batch_size=batch_size,
        margin=margin,
        objective_name=objective_name,
        best_epoch=best_epoch,
        history=history,
        validation_metrics=best_metrics,
        test_metrics=test_metrics,
        checkpoint_path=str(checkpoint_path),
    )