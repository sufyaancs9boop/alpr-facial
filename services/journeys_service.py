import uuid
from datetime import datetime
from sqlalchemy import select
from models.journey import Journey, JourneySighting


class JourneysService:
    def __init__(self, db_factory):
        self._db_factory = db_factory

    async def record_sighting(self, data: dict):
        """Upsert a journey for the plate and append a sighting."""
        plate_text = data["plateText"]
        now = datetime.utcnow()
        async with self._db_factory() as db:
            result = await db.execute(
                select(Journey)
                .where(Journey.plateText == plate_text)
                .where(Journey.status == "active")
                .order_by(Journey.startedAt.desc())
            )
            journey = result.scalar_one_or_none()
            if not journey:
                journey = Journey(
                    plateText=plate_text,
                    status="active",
                    startedAt=now,
                    lastSeenAt=now,
                )
                db.add(journey)
                await db.flush()
            else:
                journey.lastSeenAt = now

            sighting = JourneySighting(
                journeyId=journey.id,
                detectionEventId=data.get("detectionEventId"),
                cameraId=data.get("cameraId"),
                cameraName=data.get("cameraName"),
                thumbnailBase64=data.get("thumbnailBase64"),
                confidence=data.get("confidence", 0),
                timestamp=now,
            )
            db.add(sighting)
            await db.commit()
