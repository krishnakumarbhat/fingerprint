from dataclasses import replace
from pathlib import Path

from src.p01_config import AppConfig
from src.p03_dataset_index import build_dataset_index
from src.p04_phase_one_split import split_records
from src.p09_training import train_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _take_identities(records, limit: int):
    selected = []
    seen = set()
    for record in records:
        if record.identity_key not in seen and len(seen) >= limit:
            continue
        seen.add(record.identity_key)
        selected.append(record)
        if len(seen) >= limit and len(selected) >= limit * 2:
            pass
    return [record for record in selected if record.identity_key in seen]


def test_train_experiment_runs_on_small_real_subset(tmp_path: Path) -> None:
    config = AppConfig.from_root(PROJECT_ROOT)
    records = build_dataset_index(config.dataset_root)
    splits = split_records(records, config.split)
    train_records = _take_identities(splits.train, 16)
    validation_records = _take_identities(splits.validation, 8)
    test_records = _take_identities(splits.test, 8)
    runtime = replace(config.runtime, batch_size=8, eval_batch_size=16, num_workers=0)
    training = replace(
        config.training,
        samples_per_identity=2,
        early_stopping_patience=1,
        max_impostor_pairs_per_identity=16,
    )
    result = train_experiment(
        model_name="compact_siamese",
        preprocess_variant="raw",
        augmentation_variant="none",
        train_records=train_records,
        validation_records=validation_records,
        test_records=test_records,
        runtime_config=runtime,
        training_config=training,
        split_config=config.split,
        learning_rate=1e-3,
        batch_size=8,
        margin=0.75,
        epochs=1,
        steps_per_epoch=2,
        checkpoint_dir=tmp_path,
    )
    assert result.history
    assert 0.0 <= result.validation_metrics.eer <= 1.0
    assert 0.0 <= result.test_metrics.eer <= 1.0
    assert Path(result.checkpoint_path).exists()