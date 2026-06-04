from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.p01_config import AppConfig
from src.p12_experiment_runner import build_profile_config, run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="SOCOFing fingerprint verification pipeline")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root that contains archive/SOCOFing.",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "balanced", "full"),
        default="balanced",
        help="Run a smoke validation, the balanced 4GB GPU profile, or a longer full profile.",
    )
    args = parser.parse_args()
    config = build_profile_config(AppConfig.from_root(args.project_root), args.profile)
    summary = run_full_pipeline(config)
    print(json.dumps({
        "best_model": summary["best_final_result"]["model_name"],
        "best_preprocess": summary["best_final_result"]["preprocess_variant"],
        "best_augmentation": summary["best_augmentation"]["augmentation_variant"],
        "best_test_eer": summary["best_final_result"]["test_metrics"]["eer"],
        "summary_path": str(config.output_root / "summaries" / "pipeline_summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()