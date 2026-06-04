from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.p03_dataset_index import FingerprintRecord


@dataclass(frozen=True)
class VerificationMetrics:
    eer: float
    threshold: float
    accuracy: float
    far: float
    frr: float
    genuine_pairs: int
    impostor_pairs: int
    roc_far: list[float]
    roc_tpr: list[float]


def build_verification_scores(
    embeddings: np.ndarray,
    records: list[FingerprintRecord],
    max_impostor_pairs_per_identity: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    by_identity: dict[str, list[tuple[FingerprintRecord, np.ndarray]]] = {}
    for embedding, record in zip(embeddings, records):
        by_identity.setdefault(record.identity_key, []).append((record, embedding))
    enrollments: dict[str, np.ndarray] = {}
    genuine_scores: list[float] = []
    rng = np.random.default_rng(seed)
    for identity_key, items in by_identity.items():
        real_indices = [index for index, (record, _) in enumerate(items) if record.source == "real"]
        anchor_index = real_indices[0] if real_indices else 0
        enrollments[identity_key] = items[anchor_index][1]
        for index, (_, embedding) in enumerate(items):
            if index == anchor_index:
                continue
            genuine_scores.append(float(np.linalg.norm(enrollments[identity_key] - embedding)))
    impostor_scores: list[float] = []
    identity_keys = sorted(enrollments)
    for identity_key in identity_keys:
        other_keys = [candidate for candidate in identity_keys if candidate != identity_key]
        if not other_keys:
            continue
        sample_size = min(max_impostor_pairs_per_identity, len(other_keys))
        for other_key in rng.choice(other_keys, size=sample_size, replace=False).tolist():
            impostor_scores.append(float(np.linalg.norm(enrollments[identity_key] - enrollments[other_key])))
    return np.asarray(genuine_scores, dtype=np.float32), np.asarray(impostor_scores, dtype=np.float32)


def compute_verification_metrics(
    embeddings: np.ndarray,
    records: list[FingerprintRecord],
    max_impostor_pairs_per_identity: int,
    seed: int,
) -> VerificationMetrics:
    genuine_scores, impostor_scores = build_verification_scores(
        embeddings=embeddings,
        records=records,
        max_impostor_pairs_per_identity=max_impostor_pairs_per_identity,
        seed=seed,
    )
    all_scores = np.concatenate((genuine_scores, impostor_scores), axis=0)
    thresholds = np.linspace(float(all_scores.min()), float(all_scores.max()), num=512, dtype=np.float32)
    far_curve = np.asarray([(impostor_scores <= threshold).mean() for threshold in thresholds], dtype=np.float32)
    frr_curve = np.asarray([(genuine_scores > threshold).mean() for threshold in thresholds], dtype=np.float32)
    differences = np.abs(far_curve - frr_curve)
    index = int(differences.argmin())
    threshold = float(thresholds[index])
    far = float(far_curve[index])
    frr = float(frr_curve[index])
    eer = float((far + frr) / 2.0)
    correct_genuine = float((genuine_scores <= threshold).mean())
    correct_impostor = float((impostor_scores > threshold).mean())
    accuracy = (correct_genuine * len(genuine_scores) + correct_impostor * len(impostor_scores)) / max(
        len(genuine_scores) + len(impostor_scores),
        1,
    )
    return VerificationMetrics(
        eer=eer,
        threshold=threshold,
        accuracy=float(accuracy),
        far=far,
        frr=frr,
        genuine_pairs=int(len(genuine_scores)),
        impostor_pairs=int(len(impostor_scores)),
        roc_far=far_curve.tolist(),
        roc_tpr=(1.0 - frr_curve).tolist(),
    )