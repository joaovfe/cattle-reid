"""
Métricas de contagem: erro absoluto, MAE, útil para comparar contagem estimada vs real.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CountMetrics:
    """Métricas de contagem para comparação."""

    predicted_count: int
    ground_truth_count: int | None
    absolute_error: int | None  # |pred - gt|
    mean_absolute_error: float | None  # para múltiplos vídeos/frames
    unique_tracklets: int
    unique_identified: int  # tracklets com animal_id resolvido


def compute_count_metrics(
    unique_tracklets: int,
    unique_identified: int,
    ground_truth_count: int | None = None,
) -> CountMetrics:
    """
    predicted_count = unique_tracklets (número de indivíduos detectados/trackeados).
    Se ground_truth_count for fornecido, calcula absolute_error.
    """
    predicted_count = unique_tracklets
    abs_err = None
    mae = None
    if ground_truth_count is not None:
        abs_err = abs(predicted_count - ground_truth_count)
        mae = float(abs_err)

    return CountMetrics(
        predicted_count=predicted_count,
        ground_truth_count=ground_truth_count,
        absolute_error=abs_err,
        mean_absolute_error=mae,
        unique_tracklets=unique_tracklets,
        unique_identified=unique_identified,
    )
