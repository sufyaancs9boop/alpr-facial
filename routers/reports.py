from fastapi import APIRouter, Query
from sqlalchemy import select, func
from typing import Optional
from datetime import datetime

from dependencies import get_db_factory
from models.detection_event import DetectionEvent
from models.watchlist import Alert

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/stats")
async def get_stats():
    db_factory = get_db_factory()
    async with db_factory() as db:
        total = (await db.execute(select(func.count()).select_from(DetectionEvent))).scalar()
        unacked_alerts = (await db.execute(
            select(func.count()).select_from(Alert).where(Alert.acknowledged == False)
        )).scalar()
        return {"totalDetections": total, "unacknowledgedAlerts": unacked_alerts}


@router.get("/top-plates")
async def top_plates(limit: int = Query(10, le=50)):
    db_factory = get_db_factory()
    async with db_factory() as db:
        result = await db.execute(
            select(DetectionEvent.plateText, func.count().label("count"))
            .group_by(DetectionEvent.plateText)
            .order_by(func.count().desc())
            .limit(limit)
        )
        return [{"plateText": row[0], "count": row[1]} for row in result]


@router.get("/vehicle-colors")
async def vehicle_colors():
    db_factory = get_db_factory()
    async with db_factory() as db:
        result = await db.execute(
            select(DetectionEvent.vehicleColor, func.count().label("count"))
            .where(DetectionEvent.vehicleColor != None)
            .group_by(DetectionEvent.vehicleColor)
            .order_by(func.count().desc())
        )
        return [{"color": row[0], "count": row[1]} for row in result]


@router.get("/source-breakdown")
async def source_breakdown():
    db_factory = get_db_factory()
    async with db_factory() as db:
        result = await db.execute(
            select(DetectionEvent.source, func.count().label("count"))
            .group_by(DetectionEvent.source)
        )
        return [{"source": row[0], "count": row[1]} for row in result]
