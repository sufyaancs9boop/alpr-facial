import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Request
from fastapi.responses import StreamingResponse
import httpx

from config import settings
from dependencies import get_alpr_service
from schemas import DetectURLRequest, DetectStreamRequest, DetectionResponse
from utils.validation import check_image_dimensions, check_video_size

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/alpr", tags=["ALPR"])


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "open-source (fast-alpr + InsightFace + YOLOv8)",
        "capabilities": {
            "lpr": True,
            "face": settings.ENABLE_FACE_DETECTION,
            "vehicle": settings.ENABLE_VEHICLE_DETECTION,
        },
    }


@router.post("/detect", response_model=DetectionResponse)
async def detect_image(
    request: Request,
    image: UploadFile = File(...),
    session_id: Optional[str] = Query(None, alias="sessionId"),
    camera_id: Optional[str] = Query(None, alias="cameraId"),
    camera_name: Optional[str] = Query(None, alias="cameraName"),
    camera_region: Optional[str] = Query(None, alias="cameraRegion"),
    thumbnail: bool = Query(True),
    frame_step: int = Query(5, alias="frameStep"),
):
    content_type = image.content_type or ""
    if not any(t in content_type for t in ("image/jpeg", "image/jpg", "image/png", "image/bmp")):
        raise HTTPException(400, "Unsupported image type")

    data = await image.read()
    if len(data) > settings.max_file_bytes:
        raise HTTPException(413, f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit")
    # Validate image dimensions
    check_image_dimensions(data)

    alpr = get_alpr_service()
    result = await alpr.detect_from_bytes(
        data, session_id=session_id,
        camera_id=camera_id, camera_name=camera_name,
        camera_region=camera_region,
        generate_thumbnail=thumbnail,
    )
    return _result_to_dict(result)


@router.post("/detect-url", response_model=DetectionResponse)
async def detect_url(body: DetectURLRequest):
    url = body.imageUrl
    if not url:
        raise HTTPException(400, "imageUrl required")
    _validate_url(url)
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            raise HTTPException(400, f"Could not fetch URL: HTTP {resp.status_code}")
        data = resp.content
    if len(data) > settings.max_file_bytes:
        raise HTTPException(413, "Remote image exceeds size limit")
    check_image_dimensions(data)

    alpr = get_alpr_service()
    result = await alpr.detect_from_bytes(
        data,
        generate_thumbnail=body.thumbnail, # type: ignore
        camera_region=body.cameraRegion,
    )
    return _result_to_dict(result)


@router.post("/detect-video")
async def detect_video(
    video: UploadFile = File(...),
    frame_step: int = Query(5, alias="frameStep"),
    thumbnail: bool = Query(True),
    camera_region: Optional[str] = Query(None, alias="cameraRegion"),
):
    data = await video.read()
    # Validate video size
    check_video_size(data)
    alpr = get_alpr_service()

    async def event_stream():
        # Disconnect idle/slow clients: allow up to 2 consecutive keepalive timeouts (~60s)
        consecutive_timeouts = 0
        try:
            async for frame in alpr.detect_video_stream(
                data,
                frame_step=frame_step,
                camera_region=camera_region,
            ):
                # reset on real data
                consecutive_timeouts = 0
                yield f"event: detection\ndata: {json.dumps(frame)}\n\n"
        except Exception as exc:
            logger.error("Video detection error: %s", exc)
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/detect-stream", response_model=DetectionResponse)
async def detect_stream(body: DetectStreamRequest):
    url = body.url
    if not url:
        raise HTTPException(400, "url required")
    frame_step = int(body.frameStep or 5)
    alpr = get_alpr_service()

    async def event_stream():
        consecutive_timeouts = 0
        try:
            async for frame in alpr.detect_live_stream(
                url,
                frame_step=frame_step,
                camera_region=body.cameraRegion,
            ):
                consecutive_timeouts = 0
                yield f"event: detection\ndata: {json.dumps(frame)}\n\n"
        except Exception as exc:
            logger.error("Stream detection error: %s", exc)
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/sessions/{session_id}/flush")
async def flush_session(session_id: str):
    alpr = get_alpr_service()
    plates = await alpr.flush_video_session(session_id)
    return {"committed": len(plates), "plates": [alpr._plate_out_to_dict(p) for p in plates]}


def _result_to_dict(result) -> dict:
    from services.alpr_service import AlprService
    return {
        "success": result.success,
        "count": result.count,
        "plates": [AlprService._plate_out_to_dict(p) for p in result.plates],
        "faces": [AlprService._face_out_to_dict(f) for f in result.faces],
        "vehicles": [AlprService._vehicle_out_to_dict(v) for v in result.vehicles],
        "processingTimeMs": result.processing_time_ms,
    }


def _validate_url(url: str):
    import re
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
    except Exception:
        raise HTTPException(400, "Invalid URL")
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, "Only HTTP/HTTPS URLs are allowed")
    host = parsed.hostname or ""
    blocked = {"localhost", "0.0.0.0", "169.254.169.254", "100.100.100.200"}
    if (host in blocked
            or re.match(r"^127\.", host)
            or re.match(r"^10\.", host)
            or re.match(r"^172\.(1[6-9]|2\d|3[01])\.", host)
            or re.match(r"^192\.168\.", host)):
        raise HTTPException(400, "Private/loopback addresses not allowed")
