from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.p01_config import AppConfig, RuntimeConfig, TrainingConfig
from src.p03_dataset_index import build_dataset_index
from src.p04_phase_one_split import save_split_manifest, split_records
from src.p09_training import ExperimentResult, train_experiment
from src.p11_visualization import (
    save_augmentation_artifacts,
    save_eer_comparison,
    save_loss_curve,
    save_preprocess_artifacts,
    save_roc_curve,
    save_search_comparison,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sort_results(results: list[ExperimentResult]) -> list[ExperimentResult]:
    return sorted(results, key=lambda item: (item.test_metrics.eer, -item.test_metrics.accuracy, item.model_name))


def _summarize_result(result: ExperimentResult) -> dict[str, Any]:
    return result.to_dict()


def build_profile_config(config: AppConfig, profile: str) -> AppConfig:
    if profile == "balanced":
        return config
    if profile == "smoke":
        return replace(
            config,
            runtime=replace(config.runtime, batch_size=16, eval_batch_size=32, num_workers=0),
            training=replace(
                config.training,
                baseline_epochs=1,
                search_epochs=1,
                final_epochs=1,
                train_steps_per_epoch=12,
                search_steps_per_epoch=8,
                batch_sizes=(16,),
                learning_rates=(1e-3,),
                margins=(0.75,),
                early_stopping_patience=1,
                max_impostor_pairs_per_identity=32,
            ),
        )
    if profile == "full":
        return replace(
            config,
            training=replace(
                config.training,
                baseline_epochs=max(config.training.baseline_epochs, 3),
                search_epochs=max(config.training.search_epochs, 3),
                final_epochs=max(config.training.final_epochs, 3),
                train_steps_per_epoch=max(config.training.train_steps_per_epoch, 180),
                search_steps_per_epoch=max(config.training.search_steps_per_epoch, 90),
            ),
        )
    raise ValueError(f"Unknown profile: {profile}")


def run_full_pipeline(config: AppConfig) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    records = build_dataset_index(config.dataset_root)
    splits = split_records(records, config.split)
    save_split_manifest(config.output_root / "splits" / "identity_split.csv", splits)
    baseline = train_experiment(
        model_name="compact_siamese",
        preprocess_variant="raw",
        augmentation_variant="none",
        train_records=splits.train,
        validation_records=splits.validation,
        test_records=splits.test,
        runtime_config=config.runtime,
        training_config=config.training,
        split_config=config.split,
        learning_rate=config.training.learning_rates[0],
        batch_size=config.runtime.batch_size,
        margin=config.training.margins[0],
        epochs=config.training.baseline_epochs,
        steps_per_epoch=config.training.search_steps_per_epoch,
        checkpoint_dir=config.output_root / "checkpoints" / "baseline",
    )
    baseline_overfit_gap = baseline.history[-1].validation_loss - baseline.history[-1].train_loss
    augmentation_results: list[ExperimentResult] = []
    for augmentation_variant in config.grid.augmentation_variants:
        augmentation_results.append(
            train_experiment(
                model_name="compact_siamese",
                preprocess_variant="gabor",
                augmentation_variant=augmentation_variant,
                train_records=splits.train,
                validation_records=splits.validation,
                test_records=splits.test,
                runtime_config=config.runtime,
                training_config=config.training,
                split_config=config.split,
                learning_rate=config.training.learning_rates[0],
                batch_size=config.runtime.batch_size,
                margin=config.training.margins[0],
                epochs=config.training.search_epochs,
                steps_per_epoch=config.training.search_steps_per_epoch,
                checkpoint_dir=config.output_root / "checkpoints" / "augmentation_search",
            )
        )
    best_augmentation = min(augmentation_results, key=lambda item: item.validation_metrics.eer)
    hyperparameter_results: list[ExperimentResult] = []
    for learning_rate in config.training.learning_rates:
        for batch_size in config.training.batch_sizes:
            for margin in config.training.margins:
                hyperparameter_results.append(
                    train_experiment(
                        model_name="compact_siamese",
                        preprocess_variant="gabor",
                        augmentation_variant=best_augmentation.augmentation_variant,
                        train_records=splits.train,
                        validation_records=splits.validation,
                        test_records=splits.test,
                        runtime_config=replace(config.runtime, batch_size=batch_size),
                        training_config=config.training,
                        split_config=config.split,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        margin=margin,
                        epochs=config.training.search_epochs,
                        steps_per_epoch=config.training.search_steps_per_epoch,
                        checkpoint_dir=config.output_root / "checkpoints" / "hyperparameter_search",
                    )
                )
    best_hyperparameters = min(hyperparameter_results, key=lambda item: item.validation_metrics.eer)
    final_results: list[ExperimentResult] = []
    final_runtime = replace(config.runtime, batch_size=best_hyperparameters.batch_size)
    for preprocess_variant in config.grid.preprocess_variants:
        for model_name in config.grid.model_variants:
            final_results.append(
                train_experiment(
                    model_name=model_name,
                    preprocess_variant=preprocess_variant,
                    augmentation_variant=best_augmentation.augmentation_variant,
                    train_records=splits.train,
                    validation_records=splits.validation,
                    test_records=splits.test,
                    runtime_config=final_runtime,
                    training_config=config.training,
                    split_config=config.split,
                    learning_rate=best_hyperparameters.learning_rate,
                    batch_size=best_hyperparameters.batch_size,
                    margin=best_hyperparameters.margin,
                    epochs=config.training.final_epochs,
                    steps_per_epoch=config.training.train_steps_per_epoch,
                    checkpoint_dir=config.output_root / "checkpoints" / "final_grid",
                )
            )
    ranked_results = _sort_results(final_results)
    docs_assets = config.project_root / "docs" / "assets"
    sample_path = config.dataset_root / "Real" / "100__M_Left_thumb_finger.BMP"
    preprocess_assets = save_preprocess_artifacts(sample_path, docs_assets / "preprocess")
    augmentation_assets = save_augmentation_artifacts(sample_path, docs_assets / "augmentation")
    baseline_loss_curve = save_loss_curve(docs_assets / "charts" / "baseline_loss.svg", baseline)
    best_roc_curve = save_roc_curve(
        docs_assets / "charts" / "best_roc.svg",
        ranked_results[0].test_metrics,
        title="Best Model ROC Trace",
    )
    final_eer_chart = save_eer_comparison(docs_assets / "charts" / "final_eer.svg", ranked_results)
    augmentation_chart = save_search_comparison(
        docs_assets / "charts" / "augmentation_search.svg",
        augmentation_results,
        title="Augmentation Search Validation EER",
    )
    hyperparameter_chart = save_search_comparison(
        docs_assets / "charts" / "hyperparameter_search.svg",
        hyperparameter_results,
        title="Hyperparameter Search Validation EER",
    )
    summary = {
        "dataset": {
            "root": str(config.dataset_root),
            "record_count": len(records),
            "train_records": len(splits.train),
            "validation_records": len(splits.validation),
            "test_records": len(splits.test),
        },
        "baseline": _summarize_result(baseline),
        "baseline_overfit_gap": baseline_overfit_gap,
        "best_augmentation": _summarize_result(best_augmentation),
        "best_hyperparameters": _summarize_result(best_hyperparameters),
        "augmentation_search": [_summarize_result(result) for result in augmentation_results],
        "hyperparameter_search": [_summarize_result(result) for result in hyperparameter_results],
        "final_results": [_summarize_result(result) for result in ranked_results],
        "best_final_result": _summarize_result(ranked_results[0]),
        "assets": {
            "preprocess": preprocess_assets,
            "augmentation": augmentation_assets,
            "baseline_loss_curve": baseline_loss_curve,
            "best_roc_curve": best_roc_curve,
            "final_eer_chart": final_eer_chart,
            "augmentation_chart": augmentation_chart,
            "hyperparameter_chart": hyperparameter_chart,
        },
    }
    _write_json(config.output_root / "summaries" / "pipeline_summary.json", summary)
    return summary