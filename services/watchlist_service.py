import logging
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.watchlist import WatchlistEntry, Alert
from services.notifications import notifications

logger = logging.getLogger(__name__)


class WatchlistService:
    def __init__(self, db_factory):
        self._db_factory = db_factory

    async def _db(self) -> AsyncSession:
        return self._db_factory()

    async def get_all(self, active_only: bool = False) -> list[dict]:
        async with self._db_factory() as db:
            q = select(WatchlistEntry)
            if active_only:
                q = q.where(WatchlistEntry.active == True)
            result = await db.execute(q)
            return [self._to_dict(e) for e in result.scalars()]

    async def create(self, data: dict) -> dict:
        async with self._db_factory() as db:
            entry = WatchlistEntry(**data)
            db.add(entry)
            await db.commit()
            await db.refresh(entry)
            return self._to_dict(entry)

    async def update(self, entry_id: str, data: dict) -> dict | None:
        async with self._db_factory() as db:
            entry = await db.get(WatchlistEntry, entry_id)
            if not entry:
                return None
            for k, v in data.items():
                setattr(entry, k, v)
            await db.commit()
            await db.refresh(entry)
            return self._to_dict(entry)

    async def delete(self, entry_id: str) -> bool:
        async with self._db_factory() as db:
            entry = await db.get(WatchlistEntry, entry_id)
            if not entry:
                return False
            await db.delete(entry)
            await db.commit()
            return True

    async def check_and_alert(self, plate_text: str, detection_event_id: str | None,
                               thumbnail: str | None = None):
        """Create an alert if plate_text is on the active watchlist."""
        async with self._db_factory() as db:
            result = await db.execute(
                select(WatchlistEntry)
                .where(WatchlistEntry.plateText == plate_text)
                .where(WatchlistEntry.active == True)
            )
            entry = result.scalar_one_or_none()
            if not entry:
                return
            alert = Alert(
                plateText=plate_text,
                watchlistEntryId=entry.id,
                detectionEventId=detection_event_id,
                reason=entry.reason,
                thumbnailBase64=thumbnail,
                timestamp=datetime.utcnow(),
            )
            db.add(alert)
            await db.commit()
            await db.refresh(alert)
            notifications.emit_alert(self._alert_to_dict(alert))
            logger.info("ALERT: '%s' matched watchlist entry %s", plate_text, entry.id)

    async def get_alerts(self, acknowledged: bool | None = None,
                          limit: int = 50, offset: int = 0) -> list[dict]:
        async with self._db_factory() as db:
            q = select(Alert).order_by(Alert.timestamp.desc()).limit(limit).offset(offset)
            if acknowledged is not None:
                q = q.where(Alert.acknowledged == acknowledged)
            result = await db.execute(q)
            return [self._alert_to_dict(a) for a in result.scalars()]

    async def acknowledge_alert(self, alert_id: str) -> dict | None:
        async with self._db_factory() as db:
            alert = await db.get(Alert, alert_id)
            if not alert:
                return None
            alert.acknowledged = True
            await db.commit()
            await db.refresh(alert)
            return self._alert_to_dict(alert)

    async def delete_alert(self, alert_id: str) -> bool:
        async with self._db_factory() as db:
            alert = await db.get(Alert, alert_id)
            if not alert:
                return False
            await db.delete(alert)
            await db.commit()
            return True

    @staticmethod
    def _to_dict(e: WatchlistEntry) -> dict:
        return {
            "id": e.id, "plateText": e.plateText, "reason": e.reason,
            "active": e.active, "createdAt": e.createdAt.isoformat(),
        }

    @staticmethod
    def _alert_to_dict(a: Alert) -> dict:
        return {
            "id": a.id, "plateText": a.plateText,
            "watchlistEntryId": a.watchlistEntryId,
            "detectionEventId": a.detectionEventId,
            "reason": a.reason, "acknowledged": a.acknowledged,
            "timestamp": a.timestamp.isoformat(),
        }
