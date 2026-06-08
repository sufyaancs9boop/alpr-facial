import logging
from datetime import datetime, timedelta
from sqlalchemy import delete
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import settings
from models.detection_event import DetectionEvent
from models.face_event import FaceEvent
from models.watchlist import Alert

logger = logging.getLogger(__name__)
_scheduler: AsyncIOScheduler | None = None


def start_retention_scheduler(db_factory):
    global _scheduler
    if settings.RETENTION_DAYS <= 0:
        logger.info("Retention disabled (RETENTION_DAYS=0)")
        return
    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(
        lambda: _purge(db_factory),
        "interval",
        hours=24,
        id="retention_purge",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("Retention scheduler started — purging records older than %d days", settings.RETENTION_DAYS)


def stop_retention_scheduler():
    if _scheduler:
        _scheduler.shutdown(wait=False)


async def _purge(db_factory):
    cutoff = datetime.utcnow() - timedelta(days=settings.RETENTION_DAYS)
    async with db_factory() as db:
        for Model in (DetectionEvent, FaceEvent, Alert):
            ts_col = getattr(Model, "timestamp", None)
            if ts_col is None:
                continue
            result = await db.execute(delete(Model).where(ts_col < cutoff))
            if result.rowcount:
                logger.info("Retention purged %d rows from %s", result.rowcount, Model.__tablename__)
        await db.commit()
