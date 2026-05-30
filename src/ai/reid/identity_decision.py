"""
Tracklet aggregation (mean of embeddings) + identity decision via similarity threshold (open-set).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

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


def merge_duplicate_tracklets(
    aggregated_by_tid: Dict[int, np.ndarray],
    results: List[Dict[str, Any]],
    similarity_threshold: float = 0.80,
) -> Tuple[Dict[int, np.ndarray], List[Dict[str, Any]]]:
    """
    Among tracklets with animal_id=None, merge pairs with cosine similarity > threshold.
    The tracklet with more embeddings (num_embeddings) survives; duplicates are removed.

    This corrige a sobre-contagem: quando 1 vaca se parte em vários tracklets (oclusão,
    troca de track_id), cada tracklet desconhecido viraria 1 Animal novo no auto-enroll.
    Fundindo os duplicados antes do enroll, 1 vaca → 1 tracklet → 1 Animal.

    Returns updated (aggregated_by_tid, results).
    """
    # Apenas tracklets desconhecidos que têm embedding agregado disponível.
    unknown: List[Dict[str, Any]] = []
    for r in results:
        if r.get("animal_id") is not None:
            continue
        tid = r.get("track_id")
        if tid is None:
            continue
        agg = aggregated_by_tid.get(int(tid))
        if agg is None or agg.size == 0:
            continue
        unknown.append(r)

    if len(unknown) < 2:
        return aggregated_by_tid, results

    # Ordena por num_embeddings desc → tracklets "mais fortes" sobrevivem.
    unknown.sort(key=lambda r: int(r.get("num_embeddings", 0)), reverse=True)

    duplicate_tids: set[int] = set()
    for i, r_keep in enumerate(unknown):
        tid_keep = int(r_keep["track_id"])
        if tid_keep in duplicate_tids:
            continue
        emb_keep = aggregated_by_tid[tid_keep]
        for r_other in unknown[i + 1 :]:
            tid_other = int(r_other["track_id"])
            if tid_other in duplicate_tids:
                continue
            emb_other = aggregated_by_tid[tid_other]
            # Embeddings já L2-normalizados em aggregate_embeddings → dot == cosseno.
            sim = float(np.dot(emb_keep, emb_other))
            if sim > similarity_threshold:
                duplicate_tids.add(tid_other)

    if not duplicate_tids:
        return aggregated_by_tid, results

    cleaned_results = [
        r for r in results
        if r.get("track_id") is None or int(r["track_id"]) not in duplicate_tids
    ]
    cleaned_aggregated = {
        tid: emb for tid, emb in aggregated_by_tid.items() if int(tid) not in duplicate_tids
    }
    return cleaned_aggregated, cleaned_results


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
