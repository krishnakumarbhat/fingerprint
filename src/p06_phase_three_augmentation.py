from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


AUGMENTATION_VARIANTS = (
    "none",
    "rotate",
    "elastic",
    "obliterate",
    "combined",
)


def _base_grid(images: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = images.shape
    ys = torch.linspace(-1.0, 1.0, height, device=images.device, dtype=images.dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=images.device, dtype=images.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)


def random_rotation(images: torch.Tensor, max_degrees: float = 45.0) -> torch.Tensor:
    batch_size = images.shape[0]
    angles = (torch.rand(batch_size, device=images.device, dtype=images.dtype) * 2.0 - 1.0) * max_degrees
    radians = torch.deg2rad(angles)
    cosines = torch.cos(radians)
    sines = torch.sin(radians)
    theta = torch.zeros((batch_size, 2, 3), device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = cosines
    theta[:, 0, 1] = -sines
    theta[:, 1, 0] = sines
    theta[:, 1, 1] = cosines
    grid = functional.affine_grid(theta, images.size(), align_corners=False)
    return functional.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def random_elastic(images: torch.Tensor, alpha: float = 6.0) -> torch.Tensor:
    batch_size, _, height, width = images.shape
    low_res = torch.rand((batch_size, 2, 8, 8), device=images.device, dtype=images.dtype) * 2.0 - 1.0
    displacement = functional.interpolate(low_res, size=(height, width), mode="bicubic", align_corners=False)
    displacement = functional.avg_pool2d(displacement, kernel_size=5, stride=1, padding=2)
    displacement[:, 0] *= alpha / max(width, 1)
    displacement[:, 1] *= alpha / max(height, 1)
    grid = _base_grid(images) + displacement.permute(0, 2, 3, 1)
    return functional.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def random_obliteration(images: torch.Tensor, min_fraction: float = 0.10, max_fraction: float = 0.22) -> torch.Tensor:
    batch_size, _, height, width = images.shape
    augmented = images.clone()
    side_fractions = torch.rand(batch_size, device=images.device, dtype=images.dtype)
    side_fractions = min_fraction + (max_fraction - min_fraction) * side_fractions
    hole_sizes = torch.clamp((side_fractions * min(height, width)).round().to(torch.int64), min=4)
    for index in range(batch_size):
        hole_size = int(hole_sizes[index].item())
        y_limit = max(height - hole_size, 1)
        x_limit = max(width - hole_size, 1)
        y_start = int(torch.randint(0, y_limit, (1,), device=images.device).item())
        x_start = int(torch.randint(0, x_limit, (1,), device=images.device).item())
        augmented[index, :, y_start : y_start + hole_size, x_start : x_start + hole_size] = 0.0
    return augmented


def apply_augmentation_variant(images: torch.Tensor, variant: str) -> torch.Tensor:
    if variant not in AUGMENTATION_VARIANTS:
        raise ValueError(f"Unknown augmentation variant: {variant}")
    if variant == "none":
        return images
    if variant == "rotate":
        return random_rotation(images)
    if variant == "elastic":
        return random_elastic(images)
    if variant == "obliterate":
        return random_obliteration(images)
    rotated = random_rotation(images)
    elastic = random_elastic(rotated)
    return random_obliteration(elastic)