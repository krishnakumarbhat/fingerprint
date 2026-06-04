from pathlib import Path

import torch

from src.p02_bmp_reader import load_socofing_bmp
from src.p05_phase_two_preprocessing import PREPROCESS_VARIANTS, apply_preprocess_variant


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "archive" / "SOCOFing"


def test_preprocess_variants_preserve_batch_shape_and_bounds() -> None:
    samples = [
        load_socofing_bmp(DATASET_ROOT / "Real" / "100__M_Left_thumb_finger.BMP"),
        load_socofing_bmp(DATASET_ROOT / "Altered" / "Altered-Easy" / "49__M_Left_thumb_finger_CR.BMP"),
    ]
    batch = torch.stack(samples, dim=0)
    for variant in PREPROCESS_VARIANTS:
        processed = apply_preprocess_variant(batch, variant)
        assert processed.shape == batch.shape
        assert 0.0 <= float(processed.min()) <= 1.0
        assert 0.0 <= float(processed.max()) <= 1.0