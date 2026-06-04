import numpy as np

from src.p03_dataset_index import FingerprintRecord
from src.p10_phase_five_evaluation import compute_verification_metrics


def test_verification_metrics_reward_well_separated_embeddings() -> None:
    records = [
        FingerprintRecord(path=None, subject_id=1, gender="M", hand="Left", finger="thumb", source="real", difficulty="Real", alteration="REAL"),
        FingerprintRecord(path=None, subject_id=1, gender="M", hand="Left", finger="thumb", source="altered", difficulty="Altered-Easy", alteration="CR"),
        FingerprintRecord(path=None, subject_id=2, gender="F", hand="Right", finger="index", source="real", difficulty="Real", alteration="REAL"),
        FingerprintRecord(path=None, subject_id=2, gender="F", hand="Right", finger="index", source="altered", difficulty="Altered-Easy", alteration="CR"),
    ]
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [0.02, 0.01],
            [2.0, 2.0],
            [2.02, 2.01],
        ],
        dtype=np.float32,
    )
    metrics = compute_verification_metrics(
        embeddings=embeddings,
        records=records,
        max_impostor_pairs_per_identity=4,
        seed=42,
    )
    assert metrics.eer < 0.05
    assert metrics.accuracy > 0.95