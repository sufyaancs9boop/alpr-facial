"""
Unified inference service — orchestrates plate, face, and vehicle ML models.
Replaces roc.service.ts as the single ML entry point.
"""
import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

import numpy as np

from config import settings
from ml.plate_detector import PlateDetector, PlateResult, passes_pre_filters, normalize_plate
from ml.face_analyzer import FaceAnalyzer, FaceResult
from ml.vehicle_classifier import VehicleClassifier, VehicleResult

logger = logging.getLogger(__name__)

_plate_detector = PlateDetector()
_face_analyzer = FaceAnalyzer(models_dir=settings.MODELS_DIR)
_vehicle_classifier = VehicleClassifier()


def get_plate_detector() -> PlateDetector:
    return _plate_detector


def get_face_analyzer() -> FaceAnalyzer:
    return _face_analyzer


def get_vehicle_classifier() -> VehicleClassifier:
    return _vehicle_classifier


async def detect_image(
    image_bytes: bytes,
    generate_thumbnail: bool = True,
) -> tuple[list[PlateResult], list[FaceResult], list[VehicleResult]]:
    """Run all detectors concurrently on a single image. Returns (plates, faces, vehicles)."""
    loop = asyncio.get_event_loop()

    tasks = [
        loop.run_in_executor(None, _plate_detector.detect, image_bytes, generate_thumbnail),
        loop.run_in_executor(None, _face_analyzer.detect, image_bytes, generate_thumbnail)
        if settings.ENABLE_FACE_DETECTION else asyncio.coroutine(lambda: [])(),
        loop.run_in_executor(None, _vehicle_classifier.detect, image_bytes, generate_thumbnail)
        if settings.ENABLE_VEHICLE_DETECTION else asyncio.coroutine(lambda: [])(),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    plates = results[0] if not isinstance(results[0], Exception) else []
    faces = results[1] if not isinstance(results[1], Exception) else []
    vehicles = results[2] if not isinstance(results[2], Exception) else []

    if isinstance(results[0], Exception):
        logger.error("Plate detection failed: %s", results[0])
    if isinstance(results[1], Exception):
        logger.error("Face detection failed: %s", results[1])
    if isinstance(results[2], Exception):
        logger.error("Vehicle detection failed: %s", results[2])

    return plates, faces, vehicles


async def detect_video_frames(
    video_bytes: bytes,
    frame_step: int = 5,
    generate_thumbnail: bool = True,
) -> AsyncGenerator[tuple[int, list[PlateResult], list[FaceResult], list[VehicleResult]], None]:
    """Yield (frame_index, plates, faces, vehicles) for each processed frame of a video."""
    import cv2
    import tempfile
    import os

    # Write to temp file so OpenCV can demux properly
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        frame_index = 0
        processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_step == 0:
                ret2, jpeg = cv2.imencode(".jpg", frame)
                if ret2:
                    image_bytes = jpeg.tobytes()
                    plates, faces, vehicles = await detect_image(image_bytes, generate_thumbnail)
                    yield processed, plates, faces, vehicles
                    processed += 1
            frame_index += 1
        cap.release()
    finally:
        os.unlink(tmp_path)


async def detect_stream_frames(
    url: str,
    frame_step: int = 5,
    generate_thumbnail: bool = True,
    should_continue=None,
) -> AsyncGenerator[tuple[int, list[PlateResult], list[FaceResult], list[VehicleResult]], None]:
    """Yield (frame_index, plates, faces, vehicles) for a live RTSP/HTTP stream."""
    import cv2
    loop = asyncio.get_event_loop()

    cap = await loop.run_in_executor(None, cv2.VideoCapture, url)
    frame_index = 0
    processed = 0
    try:
        while True:
            if should_continue and not should_continue():
                break
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                logger.warning("Stream ended or failed: %s", url)
                break
            if frame_index % frame_step == 0:
                ret2, jpeg = cv2.imencode(".jpg", frame)
                if ret2:
                    image_bytes = jpeg.tobytes()
                    plates, faces, vehicles = await detect_image(image_bytes, generate_thumbnail)
                    yield processed, plates, faces, vehicles
                    processed += 1
            frame_index += 1
    finally:
        cap.release()
