"""
GET /events: filter by animal_id, time, type.
"""
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import select

from core.database.session import async_session_factory
from core.database.models import Event

router = APIRouter()


class EventResponse(BaseModel):
    id: int
    animal_id: int | None
    tracklet_id: int | None
    event_type: str
    payload: dict[str, Any] | None
    timestamp: datetime


@router.get("", response_model=list[EventResponse])
async def list_events(
    animal_id: int | None = Query(None),
    event_type: str | None = Query(None),
    since: datetime | None = Query(None),
    limit: int = Query(100, le=500),
):
    async with async_session_factory() as session:
        q = select(Event).order_by(Event.timestamp.desc()).limit(limit)
        if animal_id is not None:
            q = q.where(Event.animal_id == animal_id)
        if event_type:
            q = q.where(Event.event_type == event_type)
        if since is not None:
            q = q.where(Event.timestamp >= since)
        result = await session.execute(q)
        events = result.scalars().all()
        return [
            EventResponse(
                id=e.id,
                animal_id=e.animal_id,
                tracklet_id=e.tracklet_id,
                event_type=e.event_type,
                payload=e.payload,
                timestamp=e.timestamp,
            )
            for e in events
        ]
