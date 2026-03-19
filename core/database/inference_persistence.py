"""
Persistência dos resultados de inferência: tracklets, eventos de contagem e métricas.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from core.database.models import AnimalCrop, Event, Tracklet


def _track_id_to_frame_range(frame_detections: dict[int, list[tuple[list[float], int]]]) -> dict[int, tuple[int, int]]:
    """Para cada track_id, (start_frame, end_frame)."""
    by_track: dict[int, list[int]] = {}
    for frame_idx, dets in frame_detections.items():
        for _bbox, track_id in dets:
            by_track.setdefault(track_id, []).append(frame_idx)
    return {
        tid: (min(frames), max(frames))
        for tid, frames in by_track.items()
    }


async def save_inference_results(
    session: AsyncSession,
    video_source: str,
    tracklet_results: list[dict[str, Any]],
    frame_detections: dict[int, list[tuple[list[float], int]]],
    metrics: dict[str, Any] | None = None,
    animal_crops: list[dict[str, Any]] | None = None,
    extra_events: list[dict[str, Any]] | None = None,
) -> list[int]:
    """
    Salva tracklets e evento de contagem no banco.
    tracklet_results: [{"track_id", "animal_id", "score", "num_embeddings"}, ...]
    frame_detections: {frame_idx: [(bbox, track_id), ...]}
    metrics: opcional, dict com precision, recall, f1, count, etc.
    Retorna lista de IDs dos tracklets criados.
    """
    track_ranges = _track_id_to_frame_range(frame_detections)
    tracklet_ids: list[int] = []
    for r in tracklet_results:
        tid = r.get("track_id")
        if tid is None:
            continue
        start_frame, end_frame = track_ranges.get(tid, (0, 0))
        t = Tracklet(
            video_source=video_source,
            start_frame=start_frame,
            end_frame=end_frame,
            track_id=tid,
            resolved_animal_id=r.get("animal_id"),
        )
        session.add(t)
        await session.flush()
        tracklet_ids.append(t.id)

    if tracklet_results or metrics:
        unique_tracklets = len(tracklet_results)
        unique_identified = sum(1 for r in tracklet_results if r.get("animal_id") is not None)
        payload: dict[str, Any] = {
            "video_source": video_source,
            "unique_tracklets": unique_tracklets,
            "unique_identified": unique_identified,
            "count": unique_tracklets,
        }
        if metrics:
            payload["metrics"] = metrics

        event = Event(
            event_type="count_summary",
            payload=payload,
            timestamp=datetime.utcnow(),
        )
        session.add(event)
        await session.flush()

    if animal_crops:
        for c in animal_crops:
            session.add(
                AnimalCrop(
                    animal_id=int(c["animal_id"]),
                    tracklet_id=c.get("tracklet_id"),
                    source_path=str(c["source_path"]),
                    frame_index=c.get("frame_index"),
                    bbox=c.get("bbox"),
                    metadata_=c.get("metadata"),
                )
            )

    if extra_events:
        for e in extra_events:
            session.add(
                Event(
                    event_type=str(e["event_type"]),
                    animal_id=e.get("animal_id"),
                    tracklet_id=e.get("tracklet_id"),
                    payload=e.get("payload"),
                    timestamp=e.get("timestamp") or datetime.utcnow(),
                )
            )

    return tracklet_ids
