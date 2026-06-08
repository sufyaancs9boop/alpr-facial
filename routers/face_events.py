from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from dependencies import get_db_factory
from models.face_event import FaceEvent

router = APIRouter(prefix="/face-events", tags=["FaceEvents"])


@router.get("/")
async def list_face_events(
    person_id: Optional[str] = Query(None, alias="personId"),
    camera_id: Optional[str] = Query(None, alias="cameraId"),
    spoof_only: bool = Query(False, alias="spoofOnly"),
    start_date: Optional[str] = Query(None, alias="startDate"),
    end_date: Optional[str] = Query(None, alias="endDate"),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    db_factory = get_db_factory()
    async with db_factory() as db:
        q = select(FaceEvent).order_by(FaceEvent.timestamp.desc()).limit(limit).offset(offset)
        if person_id:
            q = q.where(FaceEvent.personId == person_id)
        if camera_id:
            q = q.where(FaceEvent.cameraId == camera_id)
        if spoof_only:
            q = q.where(FaceEvent.spoofDetected == True)
        if start_date:
            q = q.where(FaceEvent.timestamp >= datetime.fromisoformat(start_date))
        if end_date:
            q = q.where(FaceEvent.timestamp <= datetime.fromisoformat(end_date))
        result = await db.execute(q)
        return [_to_dict(e) for e in result.scalars()]


@router.delete("/{event_id}")
async def delete_face_event(event_id: str):
    db_factory = get_db_factory()
    async with db_factory() as db:
        ev = await db.get(FaceEvent, event_id)
        if not ev:
            raise HTTPException(404, "FaceEvent not found")
        await db.delete(ev)
        await db.commit()
    return {"ok": True}


def _to_dict(e: FaceEvent) -> dict:
    return {
        "id": e.id, "personId": e.personId, "personName": e.personName,
        "confidence": e.confidence, "quality": e.quality,
        "spoofScore": e.spoofScore, "spoofDetected": e.spoofDetected,
        "occluded": e.occluded, "thumbnailBase64": e.thumbnailBase64,
        "boundingBox": {"x": e.x, "y": e.y, "width": e.width, "height": e.height},
        "cameraId": e.cameraId, "cameraName": e.cameraName,
        "detectionEventId": e.detectionEventId,
        "timestamp": e.timestamp.isoformat(),
    }
