from pathlib import Path

from src.p01_config import AppConfig
from src.p03_dataset_index import build_dataset_index
from src.p04_phase_one_split import split_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_split_is_identity_disjoint_and_ratio_aligned() -> None:
    config = AppConfig.from_root(PROJECT_ROOT)
    records = build_dataset_index(config.dataset_root)
    splits = split_records(records, config.split)
    train_ids = {record.identity_key for record in splits.train}
    validation_ids = {record.identity_key for record in splits.validation}
    test_ids = {record.identity_key for record in splits.test}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    total_ids = len(train_ids | validation_ids | test_ids)
    assert abs(len(train_ids) / total_ids - config.split.train_ratio) < 0.02
    assert abs(len(validation_ids) / total_ids - config.split.validation_ratio) < 0.02
    assert abs(len(test_ids) / total_ids - config.split.test_ratio) < 0.02