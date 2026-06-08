from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from dependencies import get_db_factory
from models.journey import Journey, JourneySighting

router = APIRouter(prefix="/journeys", tags=["Journeys"])


@router.get("/")
async def list_journeys(limit: int = 50, offset: int = 0):
    db_factory = get_db_factory()
    async with db_factory() as db:
        q = (select(Journey)
             .order_by(Journey.lastSeenAt.desc())
             .limit(limit).offset(offset))
        result = await db.execute(q)
        return [_to_dict(j) for j in result.scalars()]


@router.get("/{journey_id}")
async def get_journey(journey_id: str):
    db_factory = get_db_factory()
    async with db_factory() as db:
        q = (select(Journey)
             .where(Journey.id == journey_id)
             .options(selectinload(Journey.sightings)))
        result = await db.execute(q)
        journey = result.scalar_one_or_none()
        if not journey:
            raise HTTPException(404, "Journey not found")
        return _to_dict_with_sightings(journey)


def _to_dict(j: Journey) -> dict:
    return {
        "id": j.id, "plateText": j.plateText, "status": j.status,
        "startedAt": j.startedAt.isoformat(), "lastSeenAt": j.lastSeenAt.isoformat(),
    }


def _to_dict_with_sightings(j: Journey) -> dict:
    d = _to_dict(j)
    d["sightings"] = [
        {
            "id": s.id, "journeyId": s.journeyId,
            "detectionEventId": s.detectionEventId,
            "cameraId": s.cameraId, "cameraName": s.cameraName,
            "confidence": s.confidence,
            "timestamp": s.timestamp.isoformat(),
        }
        for s in (j.sightings or [])
    ]
    return d
