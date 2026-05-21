"""GET /tracklets: list tracklets with posture classification for the frontend."""
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import select

from core.database.session import async_session_factory
from core.database.models import Tracklet

router = APIRouter()


class TrackletResponse(BaseModel):
    id: int
    track_id: int
    video_source: str | None
    start_frame: int
    end_frame: int
    resolved_animal_id: int | None
    posture_label: str | None
    posture_score: float | None
    created_at: datetime


@router.get("", response_model=list[TrackletResponse])
async def list_tracklets(
    video_source: str | None = Query(None),
    posture_label: str | None = Query(None),
    animal_id: int | None = Query(None),
    limit: int = Query(100, le=1000),
):
    async with async_session_factory() as session:
        q = select(Tracklet).order_by(Tracklet.created_at.desc()).limit(limit)
        if video_source:
            q = q.where(Tracklet.video_source == video_source)
        if posture_label:
            q = q.where(Tracklet.posture_label == posture_label)
        if animal_id is not None:
            q = q.where(Tracklet.resolved_animal_id == animal_id)
        result = await session.execute(q)
        tracklets = result.scalars().all()
        return [
            TrackletResponse(
                id=t.id,
                track_id=t.track_id,
                video_source=t.video_source,
                start_frame=t.start_frame,
                end_frame=t.end_frame,
                resolved_animal_id=t.resolved_animal_id,
                posture_label=t.posture_label,
                posture_score=t.posture_score,
                created_at=t.created_at,
            )
            for t in tracklets
        ]


@router.get("/summary")
async def posture_summary(video_source: str | None = Query(None)):
    """Contagem por postura — útil para dashboard do frontend."""
    async with async_session_factory() as session:
        q = select(Tracklet)
        if video_source:
            q = q.where(Tracklet.video_source == video_source)
        result = await session.execute(q)
        tracklets = result.scalars().all()

    total = len(tracklets)
    by_posture: dict[str, int] = {}
    for t in tracklets:
        label = t.posture_label or "unknown"
        by_posture[label] = by_posture.get(label, 0) + 1

    return {
        "total": total,
        "by_posture": by_posture,
    }
