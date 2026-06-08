import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from dependencies import get_watchlist_service
from services.notifications import notifications

router = APIRouter(prefix="/watchlist", tags=["Watchlist"])


@router.get("/")
async def list_watchlist(active_only: bool = Query(False, alias="activeOnly")):
    svc = get_watchlist_service()
    return await svc.get_all(active_only=active_only)


@router.post("/")
async def create_entry(body: dict):
    svc = get_watchlist_service()
    return await svc.create(body)


@router.patch("/{entry_id}")
async def update_entry(entry_id: str, body: dict):
    svc = get_watchlist_service()
    result = await svc.update(entry_id, body)
    if not result:
        raise HTTPException(404, "Entry not found")
    return result


@router.delete("/{entry_id}")
async def delete_entry(entry_id: str):
    svc = get_watchlist_service()
    if not await svc.delete(entry_id):
        raise HTTPException(404, "Entry not found")
    return {"ok": True}
