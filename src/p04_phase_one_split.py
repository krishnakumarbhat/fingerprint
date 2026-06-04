from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset, Sampler

from src.p01_config import SplitConfig
from src.p02_bmp_reader import load_socofing_bmp
from src.p03_dataset_index import FingerprintRecord


@dataclass(frozen=True)
class DatasetSplits:
    train: list[FingerprintRecord]
    validation: list[FingerprintRecord]
    test: list[FingerprintRecord]


def build_identity_groups(records: list[FingerprintRecord]) -> dict[str, list[FingerprintRecord]]:
    grouped: dict[str, list[FingerprintRecord]] = defaultdict(list)
    for record in records:
        grouped[record.identity_key].append(record)
    return dict(grouped)


def split_records(records: list[FingerprintRecord], split_config: SplitConfig) -> DatasetSplits:
    grouped = build_identity_groups(records)
    identity_keys = np.array(sorted(grouped))
    rng = np.random.default_rng(split_config.seed)
    shuffled = rng.permutation(identity_keys)
    total = len(shuffled)
    train_end = int(total * split_config.train_ratio)
    validation_end = train_end + int(total * split_config.validation_ratio)
    train_keys = set(shuffled[:train_end].tolist())
    validation_keys = set(shuffled[train_end:validation_end].tolist())
    test_keys = set(shuffled[validation_end:].tolist())
    train = [record for record in records if record.identity_key in train_keys]
    validation = [record for record in records if record.identity_key in validation_keys]
    test = [record for record in records if record.identity_key in test_keys]
    return DatasetSplits(train=train, validation=validation, test=test)


def build_label_map(records: list[FingerprintRecord]) -> dict[str, int]:
    keys = sorted({record.identity_key for record in records})
    return {key: index for index, key in enumerate(keys)}


class FingerprintDataset(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(
        self,
        records: list[FingerprintRecord],
        label_map: dict[str, int],
        target_size: tuple[int, int] | None = None,
    ):
        self.records = records
        self.label_map = label_map
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self.records[index]
        image = load_socofing_bmp(record.path)
        if self.target_size is not None and tuple(image.shape[-2:]) != self.target_size:
            image = functional.interpolate(
                image.unsqueeze(0),
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        label = self.label_map[record.identity_key]
        return image, label, index


class IdentityBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        records: list[FingerprintRecord],
        batch_size: int,
        samples_per_identity: int,
        steps_per_epoch: int,
        seed: int,
    ) -> None:
        if batch_size % samples_per_identity != 0:
            raise ValueError("batch_size must be divisible by samples_per_identity")
        self.batch_size = batch_size
        self.samples_per_identity = samples_per_identity
        self.identities_per_batch = batch_size // samples_per_identity
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            grouped[record.identity_key].append(index)
        self.grouped_indices = {key: np.array(value, dtype=np.int64) for key, value in grouped.items()}
        self.identity_keys = np.array(sorted(self.grouped_indices), dtype=object)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.steps_per_epoch):
            chosen = rng.choice(
                self.identity_keys,
                size=self.identities_per_batch,
                replace=len(self.identity_keys) < self.identities_per_batch,
            )
            batch: list[int] = []
            for identity_key in chosen.tolist():
                indices = self.grouped_indices[identity_key]
                sampled = rng.choice(
                    indices,
                    size=self.samples_per_identity,
                    replace=len(indices) < self.samples_per_identity,
                )
                batch.extend(int(index) for index in sampled.tolist())
            yield batch


def save_split_manifest(path: Path, splits: DatasetSplits) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["split,identity_key,file_name,source,difficulty,alteration"]
    for split_name, records in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        for record in records:
            lines.append(
                f"{split_name},{record.identity_key},{record.path.name},{record.source},{record.difficulty},{record.alteration}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")