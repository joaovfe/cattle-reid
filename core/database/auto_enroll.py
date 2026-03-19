from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.models import Animal, EmbeddingRef


async def auto_enroll_unknown_tracklets(
    *,
    session: AsyncSession,
    store: Any,
    tracklet_results: list[dict[str, Any]],
    tracklet_aggregated_embeddings: dict[int, np.ndarray],
    video_source: str | None = None,
) -> dict[int, int]:
    """
    For each result where animal_id is None, create a new Animal and add its embedding to FAISS.

    Returns: mapping {track_id -> new_animal_id}.
    """
    track_to_animal: dict[int, int] = {}
    for r in tracklet_results:
        tid = r.get("track_id")
        if tid is None:
            continue
        if r.get("animal_id") is not None:
            continue
        emb = tracklet_aggregated_embeddings.get(int(tid))
        if emb is None or getattr(emb, "size", 0) == 0:
            continue

        animal = Animal(
            external_id=None,
            name=None,
            metadata_={
                "enrollment": "auto",
                "video_source": video_source,
                "track_id": int(tid),
            },
        )
        session.add(animal)
        await session.flush()  # assigns animal.id

        # Add embedding to FAISS (in-memory store owned by API/CLI).
        store.add(int(animal.id), emb)
        session.add(EmbeddingRef(animal_id=int(animal.id)))

        r["animal_id"] = int(animal.id)
        r["score"] = float(r.get("score") or 0.0)
        r["enrolled"] = True
        track_to_animal[int(tid)] = int(animal.id)

    return track_to_animal

