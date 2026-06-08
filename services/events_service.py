from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from models.detection_event import DetectionEvent


class EventsService:
    def __init__(self, db_factory):
        self._db_factory = db_factory

    async def create(self, data: dict) -> dict:
        async with self._db_factory() as db:
            event = DetectionEvent(
                plateText=data["plateText"],
                confidence=data.get("confidence", 0),
                personId=data.get("personId"),
                personName=data.get("personName"),
                source=data.get("source", "image"),
                thumbnailBase64=data.get("thumbnailBase64"),
                x=data.get("x", 0),
                y=data.get("y", 0),
                width=data.get("width", 0),
                height=data.get("height", 0),
                vehicleMake=data.get("vehicleMake"),
                vehicleModel=data.get("vehicleModel"),
                vehicleColor=data.get("vehicleColor"),
                vehicleThumbnail=data.get("vehicleThumbnail"),
                direction=data.get("direction"),
                cameraId=data.get("cameraId"),
                cameraName=data.get("cameraName"),
                gunDetected=data.get("gunDetected", False),
                timestamp=datetime.utcnow(),
            )
            db.add(event)
            await db.commit()
            await db.refresh(event)
            return _to_dict(event)


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
