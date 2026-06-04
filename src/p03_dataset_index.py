from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FingerprintRecord:
    path: Path
    subject_id: int
    gender: str
    hand: str
    finger: str
    source: str
    difficulty: str
    alteration: str

    @property
    def identity_key(self) -> str:
        return f"{self.subject_id:03d}__{self.gender}_{self.hand}_{self.finger}_finger"


def parse_record(path: Path) -> FingerprintRecord:
    subject_text, rest = path.stem.split("__", maxsplit=1)
    parts = rest.split("_")
    if len(parts) < 4 or parts[3].lower() != "finger":
        raise ValueError(f"Unexpected SOCOFing filename: {path.name}")
    subject_id = int(subject_text)
    gender = parts[0]
    hand = parts[1]
    finger = parts[2]
    alteration = parts[4] if len(parts) > 4 else "REAL"
    source = "altered" if any(p.lower() == "altered" for p in path.parts) else "real"
    difficulty = path.parent.name if source == "altered" else "Real"
    return FingerprintRecord(
        path=path,
        subject_id=subject_id,
        gender=gender,
        hand=hand,
        finger=finger,
        source=source,
        difficulty=difficulty,
        alteration=alteration,
    )


def build_dataset_index(dataset_root: Path) -> list[FingerprintRecord]:
    # Find Real/real and Altered/altered directories case-insensitively
    real_root = None
    altered_root = None
    if dataset_root.exists():
        for sub in dataset_root.iterdir():
            if sub.is_dir():
                if sub.name.lower() == "real":
                    real_root = sub
                elif sub.name.lower() == "altered":
                    altered_root = sub
                    
    if real_root is None:
        real_root = dataset_root / "Real"
    if altered_root is None:
        altered_root = altered_root or (dataset_root / "Altered")
        
    # Get all BMP/bmp/Bmp files case-insensitively
    real_paths = []
    if real_root.exists():
        for pattern in ("*.BMP", "*.bmp", "*.Bmp", "*.BMP*"):
            real_paths.extend(real_root.glob(pattern))
    real_paths = sorted(list(set(real_paths)))
    
    altered_paths = []
    if altered_root.exists():
        for pattern in ("*.BMP", "*.bmp", "*.Bmp", "*.BMP*"):
            altered_paths.extend(altered_root.rglob(pattern))
    altered_paths = sorted(list(set(altered_paths)))
    
    real_records = [parse_record(path) for path in real_paths]
    altered_records = [parse_record(path) for path in altered_paths]
    records = real_records + altered_records
    if not records:
        raise FileNotFoundError(f"No SOCOFing BMP files were found under {dataset_root}")
    return records