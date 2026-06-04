from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as functional


PREPROCESS_VARIANTS = (
    "raw",
    "segmented",
    "normalized",
    "gabor",
    "skeleton",
)


def _as_batch(images: torch.Tensor) -> torch.Tensor:
    if images.dim() == 3:
        return images.unsqueeze(0)
    return images


def global_block_segmentation(images: torch.Tensor, block_size: int = 8, threshold_scale: float = 0.35) -> torch.Tensor:
    images = _as_batch(images)
    mean = functional.avg_pool2d(images, kernel_size=block_size, stride=block_size, ceil_mode=True)
    mean_square = functional.avg_pool2d(images.square(), kernel_size=block_size, stride=block_size, ceil_mode=True)
    variance = (mean_square - mean.square()).clamp_min(0.0)
    threshold = variance.mean(dim=(2, 3), keepdim=True) * threshold_scale
    coarse_mask = (variance > threshold).float()
    mask = functional.interpolate(coarse_mask, size=images.shape[-2:], mode="nearest")
    mask = functional.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)
    return (mask > 0.2).float()


def adaptive_intensity_normalization(
    images: torch.Tensor,
    mask: torch.Tensor,
    target_mean: float = 0.5,
    target_std: float = 0.22,
) -> torch.Tensor:
    images = _as_batch(images)
    mask = _as_batch(mask)
    masked_sum = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    mean = (images * mask).sum(dim=(2, 3), keepdim=True) / masked_sum
    variance = (((images - mean) * mask).square().sum(dim=(2, 3), keepdim=True) / masked_sum).clamp_min(1e-6)
    normalized = ((images - mean) / variance.sqrt()) * target_std + target_mean
    return normalized.clamp(0.0, 1.0) * mask


def estimate_ridge_orientation(images: torch.Tensor) -> torch.Tensor:
    images = _as_batch(images)
    device = images.device
    dtype = images.dtype
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    grad_x = functional.conv2d(images, sobel_x, padding=1)
    grad_y = functional.conv2d(images, sobel_y, padding=1)
    grad_xx = functional.avg_pool2d(grad_x.square(), kernel_size=9, stride=1, padding=4)
    grad_yy = functional.avg_pool2d(grad_y.square(), kernel_size=9, stride=1, padding=4)
    grad_xy = functional.avg_pool2d(grad_x * grad_y, kernel_size=9, stride=1, padding=4)
    orientation = 0.5 * torch.atan2(2.0 * grad_xy, grad_xx - grad_yy + 1e-6)
    return torch.remainder(orientation + math.pi, math.pi)


@lru_cache(maxsize=32)
def _gabor_bank(device_name: str, dtype_name: str, kernel_size: int = 11) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device_name)
    dtype = getattr(torch, dtype_name)
    sigma = 3.0
    wavelength = 5.5
    gamma = 0.6
    angles = torch.linspace(0.0, math.pi, steps=7, device=device, dtype=dtype)[:-1]
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size // 2)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernels = []
    for angle in angles:
        x_rotated = xx * torch.cos(angle) + yy * torch.sin(angle)
        y_rotated = -xx * torch.sin(angle) + yy * torch.cos(angle)
        gaussian = torch.exp(-(x_rotated.square() + (gamma * y_rotated).square()) / (2.0 * sigma * sigma))
        carrier = torch.cos(2.0 * math.pi * x_rotated / wavelength)
        kernel = gaussian * carrier
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.abs().sum().clamp_min(1e-6)
        kernels.append(kernel)
    return torch.stack(kernels).unsqueeze(1), angles


def orientation_selective_gabor(images: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    images = _as_batch(images)
    mask = _as_batch(mask)
    orientation = estimate_ridge_orientation(images)
    kernels, angles = _gabor_bank(str(images.device), str(images.dtype).split(".")[-1])
    responses = functional.conv2d(images, kernels, padding=kernels.shape[-1] // 2)
    angle_grid = angles.view(1, -1, 1, 1)
    angular_delta = torch.remainder(torch.abs(orientation - angle_grid), math.pi)
    angular_delta = torch.minimum(angular_delta, math.pi - angular_delta)
    kernel_index = angular_delta.argmin(dim=1, keepdim=True)
    selected = responses.gather(dim=1, index=kernel_index)
    masked_sum = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    mean = (selected * mask).sum(dim=(2, 3), keepdim=True) / masked_sum
    variance = (((selected - mean) * mask).square().sum(dim=(2, 3), keepdim=True) / masked_sum).clamp_min(1e-6)
    enhanced = torch.sigmoid((selected - mean) / variance.sqrt()) * mask
    return enhanced, orientation


def adaptive_binarization(images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    images = _as_batch(images)
    mask = _as_batch(mask)
    local_mean = functional.avg_pool2d(images, kernel_size=9, stride=1, padding=4)
    return ((images >= local_mean).float() * mask).clamp(0.0, 1.0)


def _thin_single(binary: np.ndarray) -> np.ndarray:
    image = binary.astype(np.uint8, copy=True)
    changed = True
    while changed:
        changed = False
        for first_pass in (True, False):
            padded = np.pad(image, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            transitions = np.zeros_like(image, dtype=np.uint8)
            for current, nxt in zip(neighbors, neighbors[1:]):
                transitions += ((current == 0) & (nxt == 1)).astype(np.uint8)
            degree = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            if first_pass:
                mask = (
                    (image == 1)
                    & (degree >= 2)
                    & (degree <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p6) == 0)
                    & ((p4 * p6 * p8) == 0)
                )
            else:
                mask = (
                    (image == 1)
                    & (degree >= 2)
                    & (degree <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p8) == 0)
                    & ((p2 * p6 * p8) == 0)
                )
            if mask.any():
                image[mask] = 0
                changed = True
    return image


def thinning(images: torch.Tensor) -> torch.Tensor:
    images = _as_batch(images)
    numpy_batch = (images[:, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    thinned = np.stack([_thin_single(sample) for sample in numpy_batch], axis=0).astype(np.float32)
    tensor = torch.from_numpy(thinned).to(device=images.device, dtype=images.dtype)
    return tensor.unsqueeze(1)


def collect_preprocess_stages(images: torch.Tensor) -> dict[str, torch.Tensor]:
    images = _as_batch(images).clamp(0.0, 1.0)
    mask = global_block_segmentation(images)
    segmented = images * mask
    normalized = adaptive_intensity_normalization(images, mask)
    gabor, orientation = orientation_selective_gabor(normalized, mask)
    binary = adaptive_binarization(gabor, mask)
    skeleton = thinning(binary)
    return {
        "raw": images,
        "segmented": segmented,
        "normalized": normalized,
        "orientation": orientation / math.pi,
        "gabor": gabor,
        "binary": binary,
        "skeleton": skeleton,
    }


def apply_preprocess_variant(images: torch.Tensor, variant: str) -> torch.Tensor:
    if variant not in PREPROCESS_VARIANTS:
        raise ValueError(f"Unknown preprocess variant: {variant}")
    stages = collect_preprocess_stages(images)
    return stages[variant]