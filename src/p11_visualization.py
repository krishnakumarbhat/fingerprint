from __future__ import annotations

from pathlib import Path

import torch
from torchvision.io import write_png

from src.p02_bmp_reader import load_socofing_bmp
from src.p05_phase_two_preprocessing import collect_preprocess_stages
from src.p06_phase_three_augmentation import AUGMENTATION_VARIANTS, apply_augmentation_variant
from src.p09_training import ExperimentResult
from src.p10_phase_five_evaluation import VerificationMetrics


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_uint8(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.dim() == 4:
        image = image[0]
    clamped = image.detach().cpu().clamp(0.0, 1.0)
    return (clamped * 255.0).round().to(torch.uint8).contiguous()


def save_png(path: Path, image: torch.Tensor) -> None:
    _ensure_parent(path)
    write_png(_to_uint8(image), str(path))


def save_strip(path: Path, images: list[torch.Tensor]) -> None:
    strip = torch.cat([_to_uint8(image).float() / 255.0 for image in images], dim=2)
    save_png(path, strip)


def save_preprocess_artifacts(sample_path: Path, output_dir: Path) -> dict[str, str]:
    image = load_socofing_bmp(sample_path)
    stages = collect_preprocess_stages(image)
    stage_paths: dict[str, str] = {}
    ordered_names = ["raw", "segmented", "normalized", "orientation", "gabor", "binary", "skeleton"]
    for name in ordered_names:
        stage_path = output_dir / f"{name}.png"
        save_png(stage_path, stages[name])
        stage_paths[name] = str(stage_path)
    save_strip(output_dir / "preprocess_strip.png", [stages[name] for name in ordered_names])
    stage_paths["strip"] = str(output_dir / "preprocess_strip.png")
    return stage_paths


def save_augmentation_artifacts(sample_path: Path, output_dir: Path) -> dict[str, str]:
    torch.manual_seed(42)
    image = load_socofing_bmp(sample_path).unsqueeze(0)
    augmentation_paths: dict[str, str] = {}
    ordered_images: list[torch.Tensor] = []
    for name in AUGMENTATION_VARIANTS:
        augmented = apply_augmentation_variant(image.clone(), name)[0]
        stage_path = output_dir / f"{name}.png"
        save_png(stage_path, augmented)
        augmentation_paths[name] = str(stage_path)
        ordered_images.append(augmented)
    save_strip(output_dir / "augmentation_strip.png", ordered_images)
    augmentation_paths["strip"] = str(output_dir / "augmentation_strip.png")
    return augmentation_paths


def _line_chart_svg(series: list[tuple[str, list[float], str]], title: str, x_label: str, y_label: str) -> str:
    width = 900
    height = 520
    left = 80
    right = 40
    top = 60
    bottom = 120
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_values = [value for _, values, _ in series for value in values]
    min_value = min(all_values)
    max_value = max(all_values)
    if max_value == min_value:
        max_value = min_value + 1.0

    def to_x(position: int, total: int) -> float:
        if total <= 1:
            return float(left)
        return left + (plot_width * position / (total - 1))

    def to_y(value: float) -> float:
        return top + plot_height * (1.0 - (value - min_value) / (max_value - min_value))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="24" font-family="Georgia">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222" stroke-width="2"/>',
        f'<text x="{width / 2}" y="{height - 30}" text-anchor="middle" font-size="18" font-family="Georgia">{x_label}</text>',
        f'<text x="24" y="{height / 2}" text-anchor="middle" font-size="18" font-family="Georgia" transform="rotate(-90 24 {height / 2})">{y_label}</text>',
    ]
    for index in range(5):
        fraction = index / 4.0
        y_value = min_value + (max_value - min_value) * fraction
        y = to_y(y_value)
        lines.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_width}" y2="{y}" stroke="#d8d0c2" stroke-width="1"/>')
        lines.append(
            f'<text x="{left - 12}" y="{y + 5}" text-anchor="end" font-size="14" font-family="monospace">{y_value:.3f}</text>'
        )
    for label, values, color in series:
        points = " ".join(
            f"{to_x(position, len(values)):.2f},{to_y(value):.2f}" for position, value in enumerate(values)
        )
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
    legend_y = top + plot_height + 40
    legend_x = left
    for index, (label, _, color) in enumerate(series):
        x = legend_x + index * 220
        lines.append(f'<rect x="{x}" y="{legend_y - 14}" width="22" height="8" fill="{color}"/>')
        lines.append(f'<text x="{x + 32}" y="{legend_y}" font-size="16" font-family="Georgia">{label}</text>')
    lines.append('</svg>')
    return "\n".join(lines)


def _bar_chart_svg(entries: list[tuple[str, float]], title: str, y_label: str) -> str:
    width = max(1000, 120 * len(entries))
    height = 620
    left = 80
    right = 40
    top = 60
    bottom = 220
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = max(value for _, value in entries) * 1.1
    if max_value <= 0:
        max_value = 1.0
    bar_width = plot_width / max(len(entries), 1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="24" font-family="Georgia">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222" stroke-width="2"/>',
        f'<text x="24" y="{height / 2}" text-anchor="middle" font-size="18" font-family="Georgia" transform="rotate(-90 24 {height / 2})">{y_label}</text>',
    ]
    colors = ["#355c7d", "#c06c84", "#f67280", "#6c5b7b", "#f8b195"]
    for index, (label, value) in enumerate(entries):
        x = left + index * bar_width + 12
        bar_height = plot_height * (value / max_value)
        y = top + plot_height - bar_height
        color = colors[index % len(colors)]
        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 24:.2f}" height="{bar_height:.2f}" fill="{color}" rx="6"/>'
        )
        lines.append(
            f'<text x="{x + (bar_width - 24) / 2:.2f}" y="{y - 10:.2f}" text-anchor="middle" font-size="12" font-family="monospace">{value:.3f}</text>'
        )
        lines.append(
            f'<text x="{x + (bar_width - 24) / 2:.2f}" y="{top + plot_height + 20:.2f}" text-anchor="end" font-size="12" font-family="monospace" transform="rotate(-50 {x + (bar_width - 24) / 2:.2f} {top + plot_height + 20:.2f})">{label}</text>'
        )
    lines.append('</svg>')
    return "\n".join(lines)


def save_text_chart(path: Path, content: str) -> None:
    _ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def save_loss_curve(path: Path, result: ExperimentResult) -> str:
    svg = _line_chart_svg(
        [
            ("train loss", [item.train_loss for item in result.history], "#355c7d"),
            ("validation loss", [item.validation_loss for item in result.history], "#c06c84"),
        ],
        title="Baseline Overfitting Check",
        x_label="Epoch",
        y_label="Loss",
    )
    save_text_chart(path, svg)
    return str(path)


def save_roc_curve(path: Path, metrics: VerificationMetrics, title: str) -> str:
    svg = _line_chart_svg(
        [("ROC", metrics.roc_tpr, "#355c7d")],
        title=title,
        x_label="Threshold Index",
        y_label="True Positive Rate",
    )
    save_text_chart(path, svg)
    return str(path)


def save_eer_comparison(path: Path, results: list[ExperimentResult]) -> str:
    entries = [
        (f"{result.preprocess_variant}:{result.model_name}", result.test_metrics.eer)
        for result in results
    ]
    svg = _bar_chart_svg(entries, title="Final Grid Test EER", y_label="Equal Error Rate")
    save_text_chart(path, svg)
    return str(path)


def save_search_comparison(path: Path, results: list[ExperimentResult], title: str) -> str:
    entries = [(result.experiment_id, result.validation_metrics.eer) for result in results]
    svg = _bar_chart_svg(entries, title=title, y_label="Validation EER")
    save_text_chart(path, svg)
    return str(path)