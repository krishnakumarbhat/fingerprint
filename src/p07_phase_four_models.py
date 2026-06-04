from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional


class L2Normalize(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return functional.normalize(inputs, p=2, dim=1)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


def build_compact_encoder(embedding_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(24),
        nn.GELU(),
        nn.MaxPool2d(2),
        nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(48),
        nn.GELU(),
        nn.MaxPool2d(2),
        nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(72),
        nn.GELU(),
        nn.MaxPool2d(2),
        nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(96),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(96, embedding_dim),
        L2Normalize(),
    )


def build_mobile_encoder(embedding_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.GELU(),
        DepthwiseSeparableBlock(16, 24, stride=2),
        DepthwiseSeparableBlock(24, 48, stride=2),
        DepthwiseSeparableBlock(48, 72, stride=2),
        DepthwiseSeparableBlock(72, 96, stride=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(96, embedding_dim),
        L2Normalize(),
    )


def create_model(model_name: str, embedding_dim: int) -> tuple[nn.Module, str]:
    if model_name == "compact_siamese":
        return build_compact_encoder(embedding_dim), "contrastive"
    if model_name == "mobile_triplet":
        return build_mobile_encoder(embedding_dim), "triplet"
    raise ValueError(f"Unknown model variant: {model_name}")