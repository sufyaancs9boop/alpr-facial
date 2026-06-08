from datetime import datetime
from models.face_event import FaceEvent


class FaceEventsService:
    def __init__(self, db_factory):
        self._db_factory = db_factory

    async def create(self, data: dict) -> dict:
        async with self._db_factory() as db:
            ev = FaceEvent(
                personId=data.get("personId"),
                personName=data.get("personName"),
                confidence=data.get("confidence", 0),
                quality=data.get("quality", 0),
                spoofScore=data.get("spoofScore"),
                spoofDetected=data.get("spoofDetected", False),
                occluded=data.get("occluded", False),
                thumbnailBase64=data.get("thumbnailBase64"),
                cameraId=data.get("cameraId"),
                cameraName=data.get("cameraName"),
                detectionEventId=data.get("detectionEventId"),
                x=data.get("x", 0),
                y=data.get("y", 0),
                width=data.get("width", 0),
                height=data.get("height", 0),
                timestamp=datetime.utcnow(),
            )
            db.add(ev)
            await db.commit()
            await db.refresh(ev)
            return {"id": ev.id, "personId": ev.personId, "timestamp": ev.timestamp.isoformat()}
