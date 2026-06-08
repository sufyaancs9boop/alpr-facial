import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import init_db, AsyncSessionLocal
from dependencies import init_dependencies
from auth import verify_api_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Initialising database…")
    await init_db()

    # Build service graph
    from services.events_service import EventsService
    from services.face_events_service import FaceEventsService
    from services.persons_service import PersonsService
    from services.watchlist_service import WatchlistService
    from services.journeys_service import JourneysService
    from services.alpr_service import AlprService
    from services.camera_worker import CameraWorkerService
    from services.retention_service import start_retention_scheduler, stop_retention_scheduler

    db_factory = AsyncSessionLocal

    events_svc = EventsService(db_factory)
    face_events_svc = FaceEventsService(db_factory)
    persons_svc = PersonsService(db_factory)
    watchlist_svc = WatchlistService(db_factory)
    journeys_svc = JourneysService(db_factory)

    alpr_svc = AlprService(
        db_factory=db_factory,
        persons_service=persons_svc,
        events_service=events_svc,
        watchlist_service=watchlist_svc,
        face_events_service=face_events_svc,
        journeys_service=journeys_svc,
    )

    camera_worker_svc = CameraWorkerService(alpr_service_factory=lambda: alpr_svc)

    init_dependencies(db_factory, alpr_svc, watchlist_svc, camera_worker_svc)

    # Pre-load face gallery from DB
    from services.inference_service import get_face_analyzer
    from sqlalchemy import select
    from models.person import Person
    async with db_factory() as db:
        result = await db.execute(select(Person))
        gallery = {
            p.id: {"name": p.name, "embeddings": p.faceEmbeddings or []}
            for p in result.scalars()
        }
    get_face_analyzer().load_gallery(gallery)
    logger.info("Face gallery loaded (%d person(s))", len(gallery))

    # Start camera workers for active cameras
    from models.camera import Camera
    async with db_factory() as db:
        from sqlalchemy import select
        result = await db.execute(select(Camera).where(Camera.active == True))
        cameras = result.scalars().all()
    for cam in cameras:
        await camera_worker_svc.start_worker(cam)
    logger.info("Started %d camera worker(s)", len(cameras))

    # Retention scheduler
    start_retention_scheduler(db_factory)

    logger.info("ALPR OSS API ready on port %d", settings.PORT)
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    await camera_worker_svc.stop_all()
    stop_retention_scheduler()
    logger.info("Shutdown complete")


app = FastAPI(
    title="ALPR OSS API",
    description="Open-source ALPR backend using fast-alpr + InsightFace + YOLOv8",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=f"/{settings.API_PREFIX}/docs",
    openapi_url=f"/{settings.API_PREFIX}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers under /api prefix, protected by optional API key
from routers import alpr, events, persons, watchlist, alerts, cameras, face_events, journeys, reports

# Public endpoint: health check (no auth required, registered before the protected router)
@app.get(f"/{settings.API_PREFIX}/alpr/health", tags=["ALPR"])
async def public_health():
    from config import settings as s
    return {
        "status": "ok",
        "engine": "open-source (fast-alpr + InsightFace + YOLOv8)",
        "capabilities": {
            "lpr": True,
            "face": s.ENABLE_FACE_DETECTION,
            "vehicle": s.ENABLE_VEHICLE_DETECTION,
            "gun": False,
        },
    }

_prefix = f"/{settings.API_PREFIX}"
_deps = [Depends(verify_api_key)]

app.include_router(alpr.router, prefix=_prefix, dependencies=_deps)
app.include_router(events.router, prefix=_prefix, dependencies=_deps)
app.include_router(persons.router, prefix=_prefix, dependencies=_deps)
app.include_router(watchlist.router, prefix=_prefix, dependencies=_deps)
app.include_router(alerts.router, prefix=_prefix, dependencies=_deps)
app.include_router(cameras.router, prefix=_prefix, dependencies=_deps)
app.include_router(face_events.router, prefix=_prefix, dependencies=_deps)
app.include_router(journeys.router, prefix=_prefix, dependencies=_deps)
app.include_router(reports.router, prefix=_prefix, dependencies=_deps)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=True)
