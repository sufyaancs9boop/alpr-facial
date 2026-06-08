import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select, delete, func, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_db_factory
from models.detection_event import DetectionEvent
from services.notifications import notifications

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/events", tags=["Events"])


@router.get("/")
async def list_events(
    plate: Optional[str] = Query(None),
    person_id: Optional[str] = Query(None, alias="personId"),
    source: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None, alias="startDate"),
    end_date: Optional[str] = Query(None, alias="endDate"),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    db_factory = get_db_factory()
    async with db_factory() as db:
        q = select(DetectionEvent).order_by(DetectionEvent.timestamp.desc())
        if plate:
            q = q.where(DetectionEvent.plateText.contains(plate.upper()))
        if person_id:
            q = q.where(DetectionEvent.personId == person_id)
        if source:
            q = q.where(DetectionEvent.source == source)
        if start_date:
            q = q.where(DetectionEvent.timestamp >= datetime.fromisoformat(start_date))
        if end_date:
            q = q.where(DetectionEvent.timestamp <= datetime.fromisoformat(end_date))
        q = q.limit(limit).offset(offset)
        result = await db.execute(q)
        return [_to_dict(e) for e in result.scalars()]


@router.delete("/{event_id}")
async def delete_event(event_id: str):
    db_factory = get_db_factory()
    async with db_factory() as db:
        event = await db.get(DetectionEvent, event_id)
        if not event:
            raise HTTPException(404, "Event not found")
        await db.delete(event)
        await db.commit()
    return {"ok": True}


@router.get("/stream")
async def stream_events():
    queue = notifications.events.subscribe()

    async def event_generator():
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"event: {msg['type']}\ndata: {json.dumps(msg['data'])}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            notifications.events.unsubscribe(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


# ── Analytics endpoints ──────────────────────────────────────────────────────

def _since(days: int) -> datetime:
    return datetime.utcnow() - timedelta(days=days)


@router.get("/summary")
async def events_summary(days: int = Query(7, ge=1, le=365)):
    db_factory = get_db_factory()
    since = _since(days)
    async with db_factory() as db:
        base = select(DetectionEvent).where(DetectionEvent.timestamp >= since)
        total = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
        unique = (await db.execute(
            select(func.count(distinct(DetectionEvent.plateText)))
            .where(DetectionEvent.timestamp >= since)
        )).scalar_one()
        avg_conf = (await db.execute(
            select(func.avg(DetectionEvent.confidence))
            .where(DetectionEvent.timestamp >= since)
        )).scalar_one()
    return {
        "total": total,
        "uniquePlates": unique,
        "avgConfidence": round(float(avg_conf), 4) if avg_conf else None,
    }


@router.get("/stats")
async def events_stats(days: int = Query(7, ge=1, le=365)):
    """Return time-bucketed detection counts. 1 day → hourly, else daily."""
    db_factory = get_db_factory()
    since = _since(days)
    async with db_factory() as db:
        rows = (await db.execute(
            select(DetectionEvent.timestamp)
            .where(DetectionEvent.timestamp >= since)
            .order_by(DetectionEvent.timestamp)
        )).scalars().all()

    if days == 1:
        buckets: dict[str, int] = {}
        for ts in rows:
            key = ts.strftime("%Y-%m-%dT%H:00:00")
            buckets[key] = buckets.get(key, 0) + 1
    else:
        buckets = {}
        for ts in rows:
            key = ts.strftime("%Y-%m-%dT00:00:00")
            buckets[key] = buckets.get(key, 0) + 1

    return [{"time": k, "count": v} for k, v in sorted(buckets.items())]


@router.get("/top-plates")
async def top_plates(limit: int = Query(10, le=50), days: int = Query(7, ge=1, le=365)):
    db_factory = get_db_factory()
    since = _since(days)
    async with db_factory() as db:
        rows = (await db.execute(
            select(DetectionEvent.plateText, func.count().label("cnt"))
            .where(DetectionEvent.timestamp >= since)
            .group_by(DetectionEvent.plateText)
            .order_by(func.count().desc())
            .limit(limit)
        )).all()
    return [{"plate": r.plateText, "count": str(r.cnt)} for r in rows]


@router.get("/top-cameras")
async def top_cameras(limit: int = Query(10, le=50), days: int = Query(7, ge=1, le=365)):
    db_factory = get_db_factory()
    since = _since(days)
    async with db_factory() as db:
        rows = (await db.execute(
            select(DetectionEvent.cameraName, func.count().label("cnt"))
            .where(DetectionEvent.timestamp >= since)
            .where(DetectionEvent.cameraName.isnot(None))
            .group_by(DetectionEvent.cameraName)
            .order_by(func.count().desc())
            .limit(limit)
        )).all()
    return [{"camera": r.cameraName, "count": str(r.cnt)} for r in rows]


@router.get("/top-persons")
async def top_persons(limit: int = Query(10, le=50)):
    db_factory = get_db_factory()
    async with db_factory() as db:
        rows = (await db.execute(
            select(DetectionEvent.personId, DetectionEvent.personName, func.count().label("cnt"))
            .where(DetectionEvent.personId.isnot(None))
            .group_by(DetectionEvent.personId, DetectionEvent.personName)
            .order_by(func.count().desc())
            .limit(limit)
        )).all()
    return [{"id": r.personId, "name": r.personName, "count": str(r.cnt)} for r in rows]


@router.get("/source-breakdown")
async def source_breakdown(days: int = Query(7, ge=1, le=365)):
    db_factory = get_db_factory()
    since = _since(days)
    async with db_factory() as db:
        rows = (await db.execute(
            select(DetectionEvent.source, func.count().label("cnt"))
            .where(DetectionEvent.timestamp >= since)
            .group_by(DetectionEvent.source)
            .order_by(func.count().desc())
        )).all()
    return [{"source": r.source, "count": str(r.cnt)} for r in rows]


@router.get("/vehicle-stats")
async def vehicle_stats(days: int = Query(7, ge=1, le=365)):
    db_factory = get_db_factory()
    since = _since(days)
    async with db_factory() as db:
        makes_rows = (await db.execute(
            select(DetectionEvent.vehicleMake, func.count().label("cnt"))
            .where(DetectionEvent.timestamp >= since)
            .where(DetectionEvent.vehicleMake.isnot(None))
            .group_by(DetectionEvent.vehicleMake)
            .order_by(func.count().desc())
            .limit(10)
        )).all()
        colors_rows = (await db.execute(
            select(DetectionEvent.vehicleColor, func.count().label("cnt"))
            .where(DetectionEvent.timestamp >= since)
            .where(DetectionEvent.vehicleColor.isnot(None))
            .group_by(DetectionEvent.vehicleColor)
            .order_by(func.count().desc())
            .limit(10)
        )).all()
    return {
        "makes": [{"make": r.vehicleMake, "count": str(r.cnt)} for r in makes_rows],
        "colors": [{"color": r.vehicleColor, "count": str(r.cnt)} for r in colors_rows],
    }


def _to_dict(e: DetectionEvent) -> dict:
    return {
        "id": e.id, "plateText": e.plateText, "confidence": e.confidence,
        "personId": e.personId, "personName": e.personName, "source": e.source,
        "thumbnailBase64": e.thumbnailBase64,
        "boundingBox": {"x": e.x, "y": e.y, "width": e.width, "height": e.height},
        "vehicleMake": e.vehicleMake, "vehicleModel": e.vehicleModel,
        "vehicleColor": e.vehicleColor, "vehicleThumbnail": e.vehicleThumbnail,
        "direction": e.direction, "cameraId": e.cameraId, "cameraName": e.cameraName,
        "gunDetected": e.gunDetected, "timestamp": e.timestamp.isoformat(),
    }
