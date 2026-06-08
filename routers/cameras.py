import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from sqlalchemy import select

from dependencies import get_db_factory, get_camera_worker_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cameras", tags=["Cameras"])

TEST_VIDEO_DIR = Path("data/test-videos")
TEST_VIDEO_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/")
async def list_cameras():
    db_factory = get_db_factory()
    worker_svc = get_camera_worker_service()
    async with db_factory() as db:
        result = await db.execute(select(__import__("models.camera", fromlist=["Camera"]).Camera))
        return [_to_dict(c, worker_svc.is_streaming(c.id)) for c in result.scalars()]


@router.post("/")
async def create_camera(body: dict):
    from models.camera import Camera
    db_factory = get_db_factory()
    worker_svc = get_camera_worker_service()
    async with db_factory() as db:
        cam = Camera(**{k: v for k, v in body.items() if k != "id"})
        db.add(cam)
        await db.commit()
        await db.refresh(cam)
        if cam.active:
            await worker_svc.start_worker(cam)
        return _to_dict(cam, worker_svc.is_streaming(cam.id))


@router.patch("/{camera_id}")
async def update_camera(camera_id: str, body: dict):
    from models.camera import Camera
    db_factory = get_db_factory()
    worker_svc = get_camera_worker_service()
    async with db_factory() as db:
        cam = await db.get(Camera, camera_id)
        if not cam:
            raise HTTPException(404, "Camera not found")
        was_active = cam.active
        for k, v in body.items():
            if hasattr(cam, k):
                setattr(cam, k, v)
        await db.commit()
        await db.refresh(cam)
    if was_active and not cam.active:
        await worker_svc.stop_worker(camera_id)
    elif not was_active and cam.active:
        await worker_svc.start_worker(cam)
    elif cam.active:
        # Restart to pick up URL or config changes
        await worker_svc.stop_worker(camera_id)
        await worker_svc.start_worker(cam)
    return _to_dict(cam, worker_svc.is_streaming(cam.id))


@router.delete("/{camera_id}")
async def delete_camera(camera_id: str):
    from models.camera import Camera
    db_factory = get_db_factory()
    worker_svc = get_camera_worker_service()
    async with db_factory() as db:
        cam = await db.get(Camera, camera_id)
        if not cam:
            raise HTTPException(404, "Camera not found")
        # Clean up test video file if present
        if cam.testVideoPath and os.path.exists(cam.testVideoPath):
            os.remove(cam.testVideoPath)
        await db.delete(cam)
        await db.commit()
    await worker_svc.stop_worker(camera_id)
    return {"ok": True}


@router.post("/{camera_id}/assign-test-video")
async def assign_test_video(camera_id: str, video: UploadFile = File(...)):
    from models.camera import Camera
    db_factory = get_db_factory()
    worker_svc = get_camera_worker_service()

    async with db_factory() as db:
        cam = await db.get(Camera, camera_id)
        if not cam:
            raise HTTPException(404, "Camera not found")

        # Remove old test video if any
        if cam.testVideoPath and os.path.exists(cam.testVideoPath):
            os.remove(cam.testVideoPath)

        # Save new video
        ext = Path(video.filename or "video.mp4").suffix or ".mp4"
        dest = TEST_VIDEO_DIR / f"{camera_id}{ext}"
        content = await video.read()
        dest.write_bytes(content)

        cam.testVideoPath = str(dest)
        await db.commit()
        await db.refresh(cam)

    # Restart worker so it picks up the test video immediately
    await worker_svc.stop_worker(camera_id)
    if cam.active:
        await worker_svc.start_worker(cam)

    logger.info("Test video assigned to camera %s → %s", cam.name, dest)
    return _to_dict(cam, worker_svc.is_streaming(cam.id))


@router.delete("/{camera_id}/assign-test-video")
async def unassign_test_video(camera_id: str):
    from models.camera import Camera
    db_factory = get_db_factory()
    worker_svc = get_camera_worker_service()

    async with db_factory() as db:
        cam = await db.get(Camera, camera_id)
        if not cam:
            raise HTTPException(404, "Camera not found")

        if cam.testVideoPath and os.path.exists(cam.testVideoPath):
            os.remove(cam.testVideoPath)
        cam.testVideoPath = None
        await db.commit()
        await db.refresh(cam)

    # Restart worker to resume RTSP URL
    await worker_svc.stop_worker(camera_id)
    if cam.active:
        await worker_svc.start_worker(cam)

    logger.info("Test video removed from camera %s — resuming %s", cam.name, cam.url)
    return _to_dict(cam, worker_svc.is_streaming(cam.id))


def _to_dict(cam, streaming: bool) -> dict:
    return {
        "id": cam.id,
        "name": cam.name,
        "url": cam.url,
        "region": cam.region,
        "frameStep": cam.frameStep,
        "active": cam.active,
        "streaming": streaming,
        "lat": cam.lat,
        "lng": cam.lng,
        "zone": cam.zone,
        "notes": cam.notes,
        "roiInclude": cam.roiInclude,
        "roiExclude": cam.roiExclude,
        "testVideoPath": cam.testVideoPath,
        "createdAt": cam.createdAt.isoformat(),
    }
