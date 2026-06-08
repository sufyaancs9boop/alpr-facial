import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from dependencies import get_watchlist_service
from services.notifications import notifications

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("/")
async def list_alerts(
    acknowledged: Optional[bool] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    svc = get_watchlist_service()
    return await svc.get_alerts(acknowledged=acknowledged, limit=limit, offset=offset)


@router.patch("/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    svc = get_watchlist_service()
    result = await svc.acknowledge_alert(alert_id)
    if not result:
        raise HTTPException(404, "Alert not found")
    return result


@router.delete("/{alert_id}")
async def delete_alert(alert_id: str):
    svc = get_watchlist_service()
    if not await svc.delete_alert(alert_id):
        raise HTTPException(404, "Alert not found")
    return {"ok": True}


@router.get("/stream")
async def stream_alerts():
    queue = notifications.alerts.subscribe()

    async def generator():
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
            notifications.alerts.unsubscribe(queue)

    return StreamingResponse(generator(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
