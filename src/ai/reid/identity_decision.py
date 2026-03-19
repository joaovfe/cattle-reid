"""
Tracklet aggregation (mean of embeddings) + identity decision via similarity threshold (open-set).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def aggregate_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    """Mean of L2-normalized embeddings, then re-normalize (BECA-style tracklet aggregation)."""
    if not embeddings:
        return np.array([])
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    mean = np.mean(arr, axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean


class IdentityDecision:
    """Open-set identity: aggregated tracklet embedding vs FAISS gallery; threshold τ."""

    def __init__(
        self,
        similarity_threshold: float = 0.0,
        min_embeddings_per_tracklet: int = 3,
        top1_top2_margin: float = 0.0,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.min_embeddings_per_tracklet = min_embeddings_per_tracklet
        self.top1_top2_margin = top1_top2_margin

    def decide(
        self,
        aggregated_embedding: np.ndarray,
        faiss_store: "FAISSStore",
        top_k: int = 5,
    ) -> Tuple[int | None, float]:
        from src.ai.reid.faiss_store import FAISSStore

        if aggregated_embedding.size == 0 or len(faiss_store) == 0:
            return None, 0.0
        candidates = faiss_store.search_animal_ids(aggregated_embedding, k=top_k)
        if not candidates:
            return None, 0.0
        best_id, best_score = candidates[0]
        if best_score < self.similarity_threshold:
            return None, best_score
        if len(candidates) > 1:
            _second_id, second_score = candidates[1]
            if (best_score - second_score) < self.top1_top2_margin:
                return None, best_score
        if best_score >= self.similarity_threshold:
            return best_id, best_score
        return None, best_score
