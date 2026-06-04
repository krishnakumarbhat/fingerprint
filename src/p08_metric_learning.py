from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class LossSummary:
    loss: torch.Tensor
    positive_distance: float
    negative_distance: float


def _distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.cdist(embeddings, embeddings, p=2)


def contrastive_batch_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> LossSummary:
    distances = _distance_matrix(embeddings)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    upper = torch.triu(torch.ones_like(same, dtype=torch.bool), diagonal=1)
    positive_mask = same & upper
    negative_mask = (~same) & upper
    positive_distances = distances[positive_mask]
    negative_distances = distances[negative_mask]
    if positive_distances.numel() == 0 or negative_distances.numel() == 0:
        zero = embeddings.sum() * 0.0
        return LossSummary(loss=zero, positive_distance=0.0, negative_distance=0.0)
    hard_negative_distances = negative_distances[negative_distances < margin]
    if hard_negative_distances.numel() == 0:
        hard_negative_distances = negative_distances
    positive_loss = positive_distances.square().mean()
    negative_loss = functional.relu(margin - hard_negative_distances).square().mean()
    loss = positive_loss + negative_loss
    return LossSummary(
        loss=loss,
        positive_distance=float(positive_distances.mean().detach().cpu().item()),
        negative_distance=float(hard_negative_distances.mean().detach().cpu().item()),
    )


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> LossSummary:
    distances = _distance_matrix(embeddings)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    same.fill_diagonal_(False)
    negative = ~same
    negative.fill_diagonal_(False)
    positive_distances = distances.masked_fill(~same, -1.0)
    negative_distances = distances.masked_fill(~negative, float("inf"))
    hardest_positive = positive_distances.max(dim=1).values
    hardest_negative = negative_distances.min(dim=1).values
    valid = same.any(dim=1) & negative.any(dim=1)
    if not torch.any(valid):
        zero = embeddings.sum() * 0.0
        return LossSummary(loss=zero, positive_distance=0.0, negative_distance=0.0)
    loss = functional.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()
    return LossSummary(
        loss=loss,
        positive_distance=float(hardest_positive[valid].mean().detach().cpu().item()),
        negative_distance=float(hardest_negative[valid].mean().detach().cpu().item()),
    )


def compute_metric_loss(
    objective_name: str,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> LossSummary:
    if objective_name == "contrastive":
        return contrastive_batch_loss(embeddings, labels, margin)
    if objective_name == "triplet":
        return batch_hard_triplet_loss(embeddings, labels, margin)
    raise ValueError(f"Unknown objective name: {objective_name}")