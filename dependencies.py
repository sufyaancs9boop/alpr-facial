"""
Application-level dependency singletons.
FastAPI doesn't have NestJS-style DI, so we use module-level singletons
assembled at startup and accessed via these getter functions.
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.alpr_service import AlprService
    from services.watchlist_service import WatchlistService
    from services.camera_worker import CameraWorkerService
    from database import AsyncSessionLocal

_db_factory = None
_alpr_service: "AlprService | None" = None
_watchlist_service: "WatchlistService | None" = None
_camera_worker_service: "CameraWorkerService | None" = None


def init_dependencies(db_factory, alpr, watchlist, camera_worker):
    global _db_factory, _alpr_service, _watchlist_service, _camera_worker_service
    _db_factory = db_factory
    _alpr_service = alpr
    _watchlist_service = watchlist
    _camera_worker_service = camera_worker


def get_db_factory():
    return _db_factory


def get_alpr_service() -> "AlprService":
    return _alpr_service


def get_watchlist_service() -> "WatchlistService":
    return _watchlist_service


def get_camera_worker_service() -> "CameraWorkerService":
    return _camera_worker_service
