"""
Métricas de avaliação de tracklets para Re-ID.

Computa estatísticas sobre animais válidos, falsos tracks (ruído),
duplicatas e contagem prevista de animais únicos.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
from collections import Counter


@dataclass
class TrackletEvaluationMetrics:
    """Métricas de avaliação de qualidade dos tracklets."""

    valid_animals: int
    false_tracks: int
    duplicate_count: int
    predicted_count: int
    total_tracklets: int

    def to_dict(self) -> dict[str, Any]:
        """Converte para dicionário serializável."""
        return asdict(self)


def compute_tracklet_evaluation_metrics(
    tracklet_results: list[dict[str, Any]],
    min_embeddings_threshold: int = 10,
) -> TrackletEvaluationMetrics:
    """
    Computa métricas de avaliação baseadas nos resultados de tracklets.

    Definições:
    -----------
    - valid_animals: tracklets onde animal_id != null AND enrolled == true AND num_embeddings >= min_embeddings_threshold
    - false_tracks: tracklets onde animal_id == null OR num_embeddings < min_embeddings_threshold
    - duplicate_count: soma de (ocorrências - 1) para cada animal_id que aparece mais de uma vez entre válidos
    - predicted_count: número de animal_id únicos válidos

    Args:
        tracklet_results: Lista de dicionários com keys: track_id, animal_id, num_embeddings, score, enrolled (opcional)
        min_embeddings_threshold: Número mínimo de embeddings para considerar tracklet válido (default: 10)

    Returns:
        TrackletEvaluationMetrics com as métricas computadas
    """
    valid_animal_ids: list[int] = []
    false_tracks = 0

    for tracklet in tracklet_results:
        animal_id = tracklet.get("animal_id")
        num_embeddings = tracklet.get("num_embeddings", 0)
        enrolled = tracklet.get("enrolled", False)

        is_false_track = (
            animal_id is None
            or num_embeddings < min_embeddings_threshold
        )

        if is_false_track:
            false_tracks += 1
        else:
            is_valid = enrolled is True and num_embeddings >= min_embeddings_threshold
            if is_valid:
                valid_animal_ids.append(animal_id)

    animal_id_counts = Counter(valid_animal_ids)
    unique_valid_ids = set(valid_animal_ids)
    duplicate_count = sum(count - 1 for count in animal_id_counts.values() if count > 1)
    predicted_count = len(unique_valid_ids)
    valid_animals = len(valid_animal_ids)

    return TrackletEvaluationMetrics(
        valid_animals=valid_animals,
        false_tracks=false_tracks,
        duplicate_count=duplicate_count,
        predicted_count=predicted_count,
        total_tracklets=len(tracklet_results),
    )
