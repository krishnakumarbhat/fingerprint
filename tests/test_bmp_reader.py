from pathlib import Path

from src.p02_bmp_reader import load_socofing_bmp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "archive" / "SOCOFing"


def test_bmp_reader_supports_observed_socofing_encodings() -> None:
    sample_paths = [
        DATASET_ROOT / "Altered" / "Altered-Easy" / "49__M_Left_thumb_finger_CR.BMP",
        DATASET_ROOT / "Real" / "100__M_Left_thumb_finger.BMP",
        DATASET_ROOT / "Real" / "226__M_Left_middle_finger.BMP",
    ]
    for path in sample_paths:
        tensor = load_socofing_bmp(path)
        assert tensor.ndim == 3
        assert tensor.shape[0] == 1
        assert 0.0 <= float(tensor.min()) <= 1.0
        assert 0.0 <= float(tensor.max()) <= 1.0