"""
Métricas de contagem: erro absoluto, MAE, útil para comparar contagem estimada vs real.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any


@dataclass
class CountMetrics:
    """Métricas de contagem para comparação."""

    predicted_count: int
    ground_truth_count: int | None
    absolute_error: int | None  # |pred - gt|
    mean_absolute_error: float | None  # para múltiplos vídeos/frames
    count_accuracy_pct: float | None  # precisão percentual de contagem (0-100)
    unique_tracklets: int
    unique_identified: int  # tracklets com animal_id resolvido


@dataclass
class TrackletEvaluationMetrics:
    """Métricas de avaliação dos tracklets para JSON de saída."""

    valid_animals: int
    false_tracks: int
    duplicate_animals: int
    predicted_count: int
    precision_pct: float


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
    count_accuracy_pct = None
    if ground_truth_count is not None:
        abs_err = abs(predicted_count - ground_truth_count)
        mae = float(abs_err)
        if ground_truth_count > 0:
            count_accuracy_pct = max(0.0, 100.0 * (1.0 - (abs_err / float(ground_truth_count))))

    return CountMetrics(
        predicted_count=predicted_count,
        ground_truth_count=ground_truth_count,
        absolute_error=abs_err,
        mean_absolute_error=mae,
        count_accuracy_pct=count_accuracy_pct,
        unique_tracklets=unique_tracklets,
        unique_identified=unique_identified,
    )


def compute_tracklet_evaluation_metrics(tracklets: list[dict[str, Any]]) -> TrackletEvaluationMetrics:
    """
    Calcula métricas de avaliação sobre os tracklets já consolidados.

    Regras:
    - valid_animals: animal_id != None, enrolled == True, num_embeddings >= 10
    - false_tracks: animal_id == None OU num_embeddings < 10
    - duplicate_animals: soma de repetições de animal_id válido (n - 1 por ID)
    - predicted_count: quantidade de animal_id válidos únicos
    """
    valid_animal_ids: list[int] = []
    false_tracks = 0

    for tracklet in tracklets:
        animal_id = tracklet.get("animal_id")
        num_embeddings = int(tracklet.get("num_embeddings") or 0)
        enrolled = tracklet.get("enrolled") is True

        is_valid = animal_id is not None and enrolled and num_embeddings >= 10
        if is_valid:
            valid_animal_ids.append(int(animal_id))

        if animal_id is None or num_embeddings < 10:
            false_tracks += 1

    counts = Counter(valid_animal_ids)
    duplicate_animals = sum(max(0, qty - 1) for qty in counts.values())
    predicted_count = len(counts)
    total_evaluated = len(valid_animal_ids) + false_tracks
    precision_pct = (100.0 * len(valid_animal_ids) / float(total_evaluated)) if total_evaluated > 0 else 0.0

    return TrackletEvaluationMetrics(
        valid_animals=len(valid_animal_ids),
        false_tracks=false_tracks,
        duplicate_animals=duplicate_animals,
        predicted_count=predicted_count,
        precision_pct=precision_pct,
    )
