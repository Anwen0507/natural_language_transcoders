"""Magnitude-preserving vector and prediction-fidelity objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from delta_nla.data import DeltaStatistics
from delta_nla.models import TargetProjection


@dataclass
class ReconstructionLosses:
    total: torch.Tensor
    vector_mse: torch.Tensor
    prediction_kl: torch.Tensor


def prediction_kl_per_example(
    true_residual: torch.Tensor,
    predicted_residual: torch.Tensor,
    projection: TargetProjection,
) -> torch.Tensor:
    with torch.no_grad():
        true_logits = projection(true_residual)
        true_log_probs = F.log_softmax(true_logits, dim=-1)
        true_probs = true_log_probs.exp()
    predicted_log_probs = F.log_softmax(projection(predicted_residual), dim=-1)
    return (true_probs * (true_log_probs - predicted_log_probs)).sum(dim=-1)


def reconstruction_losses(
    predicted_x: torch.Tensor,
    true_x: torch.Tensor,
    r0: torch.Tensor,
    stats: DeltaStatistics,
    projection: TargetProjection,
    prediction_weight: float,
) -> ReconstructionLosses:
    vector_mse = (predicted_x.float() - true_x.float()).square().mean(dim=-1)
    true_delta = stats.unstandardize(true_x.float())
    predicted_delta = stats.unstandardize(predicted_x.float())
    prediction_kl = prediction_kl_per_example(
        r0.float() + true_delta,
        r0.float() + predicted_delta,
        projection,
    )
    baseline = max(stats.prediction_kl_baseline, 1e-8)
    total = vector_mse + float(prediction_weight) * prediction_kl / baseline
    return ReconstructionLosses(total=total, vector_mse=vector_mse, prediction_kl=prediction_kl)
