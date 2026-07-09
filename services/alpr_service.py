"""
Detection orchestration — replaces alpr.service.ts.
Handles pre-filters, session management, enrichment, DB logging, alerts.
"""
import asyncio
import hashlib
import logging
import time
from typing import Optional, AsyncGenerator
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from ml.plate_detector import PlateResult, passes_pre_filters, normalize_plate
from ml.ocr_quality import (
    OcrReading,
    best_fuzzy_match,
    levenshtein,
    low_confidence_characters,
    vote_ocr_readings,
)
from ml.face_analyzer import FaceResult
from ml.vehicle_classifier import VehicleResult
from services.inference_service import detect_image, detect_video_frames, detect_stream_frames, get_plate_detector
from services.plate_tracker import (
    PlateTracker,
    PlateFeedDeduplicator,
    _PlateInput as TrackerPlate,
    _BBox as TrackerBBox,
)
from services.vehicle_tracker import VehicleTracker, _PlateInput as VTrackerPlate, _BBox as VTrackerBBox
from services.notifications import notifications

logger = logging.getLogger(__name__)

PLATE_COOLDOWN_S = 3.0
SESSION_TTL_S = 120.0


@dataclass
class PlateOut:
    text: str
    confidence: float
    quality: float
    bounding_box: dict
    thumbnail: Optional[str] = None
    char_confidences: Optional[list[float]] = None
    low_confidence_chars: Optional[list[dict]] = None
    manual_review_required: bool = False
    original_text: Optional[str] = None
    corrected: bool = False
    correction_distance: Optional[int] = None
    correction_source: Optional[str] = None
    direction: Optional[str] = None
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_color: Optional[str] = None
    vehicle_thumbnail: Optional[str] = None
    person_id: Optional[str] = None
    person_name: Optional[str] = None


@dataclass
class FaceOut:
    confidence: float
    quality: float
    bounding_box: dict
    thumbnail: Optional[str] = None
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    similarity: Optional[float] = None
    spoof_score: Optional[float] = None
    spoof_detected: bool = False
    occluded: bool = False


@dataclass
class VehicleOut:
    make: Optional[str]
    model: Optional[str]
    color: Optional[str]
    type: Optional[str]
    confidence: float
    bounding_box: dict
    thumbnail: Optional[str] = None


@dataclass
class DetectionResult:
    success: bool
    count: int
    plates: list[PlateOut]
    faces: list[FaceOut]
    vehicles: list[VehicleOut]
    processing_time_ms: int
    gun_detected: bool = False


@dataclass
class _VideoSession:
    tracker: VehicleTracker
    camera_id: Optional[str]
    camera_name: Optional[str]
    created_at: float
    watchlist_checked: set  # plate texts already alerted in this session


class AlprService:
    def __init__(self, db_factory, persons_service, events_service, watchlist_service,
                 face_events_service, journeys_service):
        self._db_factory = db_factory
        self._persons = persons_service
        self._events = events_service
        self._watchlist = watchlist_service
        self._face_events = face_events_service
        self._journeys = journeys_service
        # Shared tracker for camera/stream SSE mode (minObs=1, same as TS version)
        self._tracker = PlateTracker(commit_after_ms=8_000, max_edit_distance=2, min_observations=1)
        self._feed_trackers: dict[str, PlateFeedDeduplicator] = {}
        self._recently_logged: dict[str, float] = {}
        self._sessions: dict[str, _VideoSession] = {}

    # ── Video session management ─────────────────────────────────────────────

    def _get_or_create_session(self, session_id: str, camera_id=None, camera_name=None) -> VehicleTracker:
        if session_id in self._sessions:
            return self._sessions[session_id].tracker
        session = _VideoSession(
            tracker=VehicleTracker(idle_ms=8_000, min_readings=1),
            camera_id=camera_id,
            camera_name=camera_name,
            created_at=time.time(),
            watchlist_checked=set(),
        )
        self._sessions[session_id] = session
        return session.tracker

    async def flush_video_session(self, session_id: str) -> list[PlateOut]:
        session = self._sessions.pop(session_id, None)
        if not session:
            return []
        source = "camera" if session.camera_id else "video"
        plates_out = []
        for tp in session.tracker.flush_all():
            plate_out = self._tracker_plate_to_out(tp)
            await self._log_and_alert(plate_out, source, session.camera_id, session.camera_name)
            plates_out.append(plate_out)
        logger.info("Session %s flushed — %d vehicle(s)", session_id, len(plates_out))
        return plates_out

    # ── Primary detection entry points ───────────────────────────────────────

    async def detect_from_bytes(
        self, image_bytes: bytes, session_id: Optional[str] = None,
        camera_id: Optional[str] = None, camera_name: Optional[str] = None,
        generate_thumbnail: bool = True, camera_region: Optional[str] = None,
    ) -> DetectionResult:
        start = time.time()
        plates_raw, faces_raw, vehicles_raw = await detect_image(image_bytes, generate_thumbnail)
        elapsed_ms = int((time.time() - start) * 1000)

        best_vehicle = (max(vehicles_raw, key=lambda v: v.confidence) if vehicles_raw else None)
        plates_out = [self._enrich_plate(p, best_vehicle) for p in plates_raw]
        plates_out = await self._apply_ocr_postprocessing(plates_out, image_bytes)
        faces_out = await self._enrich_faces(faces_raw)
        vehicles_out = [self._map_vehicle(v) for v in vehicles_raw]

        if session_id:
            return await self._process_into_session(
                session_id, plates_out, faces_out, vehicles_out,
                elapsed_ms, camera_id, camera_name, camera_region,
            )

        await self._process_and_log(plates_out, faces_out, "image", camera_region)
        return DetectionResult(
            success=True, count=len(plates_out) + len(faces_out),
            plates=plates_out, faces=faces_out, vehicles=vehicles_out,
            processing_time_ms=elapsed_ms,
        )

    async def _process_into_session(
        self, session_id: str, plates: list[PlateOut], faces: list[FaceOut],
        vehicles: list[VehicleOut], elapsed_ms: int,
        camera_id=None, camera_name=None, camera_region=None,
    ) -> DetectionResult:
        tracker = self._get_or_create_session(session_id, camera_id, camera_name)
        session = self._sessions[session_id]

        valid = [p for p in plates if self._passes_pre_filters_out(p, camera_region)]
        for plate in valid:
            tp = self._out_to_tracker_plate(plate)
            tracker.observe(tp)
            if plate.text not in session.watchlist_checked:
                session.watchlist_checked.add(plate.text)
                asyncio.create_task(
                    self._watchlist.check_and_alert(plate.text, None, plate.thumbnail)
                )

        return DetectionResult(
            success=True, count=len(valid) + len(faces),
            plates=valid, faces=faces, vehicles=vehicles,
            processing_time_ms=elapsed_ms,
        )

    async def _process_and_log(
        self, plates: list[PlateOut], faces: list[FaceOut], source: str,
        camera_region: Optional[str] = None,
    ):
        valid = [p for p in plates if self._passes_pre_filters_out(p, camera_region)]
        for plate in valid:
            await self._log_and_alert(plate, source)
        if settings.PERSIST_FACE_EVENTS:
            for face in faces:
                await self._save_face_event(face)

    # ── Stream / video generators ────────────────────────────────────────────

    async def detect_video_stream(
        self, video_bytes: bytes, frame_step: int = 5, camera_region: Optional[str] = None,
    ) -> AsyncGenerator[dict, None]:
        scope_key = f"video:{uuid4().hex}"
        feed_tracker = self._get_feed_tracker(scope_key)
        try:
            async for frame_idx, plates_raw, faces_raw, vehicles_raw in detect_video_frames(
                video_bytes, frame_step
            ):
                best_vehicle = max(vehicles_raw, key=lambda v: v.confidence) if vehicles_raw else None
                plates_out = [self._enrich_plate(p, best_vehicle) for p in plates_raw]
                plates_out = await self._apply_ocr_postprocessing(plates_out, None)
                faces_out = await self._enrich_faces(faces_raw)
                vehicles_out = [self._map_vehicle(v) for v in vehicles_raw]
                feed_plate_events: list[dict] = []
                for plate in plates_out:
                    tracker_plate = self._out_to_tracker_plate(plate)
                    committed_feed = feed_tracker.observe(frame_idx, tracker_plate) or []
                    for winner in committed_feed:
                        feed_plate_events.append(
                            self._plate_out_to_dict(self._tracker_plate_to_out(winner))
                        )
                    if not self._passes_pre_filters_out(plate, camera_region):
                        continue
                    committed = self._tracker.observe(tracker_plate) or []
                    for winner in committed:
                        await self._log_committed(winner, "video")
                if settings.PERSIST_FACE_EVENTS:
                    for face in faces_out:
                        await self._save_face_event(face)
                yield {
                    "frameIndex": frame_idx,
                    "plates": [self._plate_out_to_dict(p) for p in plates_out],
                    "feedPlateEvents": feed_plate_events,
                    "feedPlates": [
                        self._plate_out_to_dict(self._tracker_plate_to_out(p))
                        for p in feed_tracker.get_feed_state()
                    ],
                    "faces": [self._face_out_to_dict(f) for f in faces_out],
                    "vehicles": [self._vehicle_out_to_dict(v) for v in vehicles_out],
                    "gunDetected": False,
                }
            for winner in self._tracker.flush_all():
                await self._log_committed(winner, "video")
        finally:
            self._feed_trackers.pop(scope_key, None)

    async def detect_live_stream(
        self, url: str, frame_step: int = 5,
        camera_id: Optional[str] = None, camera_name: Optional[str] = None,
        camera_region: Optional[str] = None, should_continue=None,
    ) -> AsyncGenerator[dict, None]:
        source = "camera" if camera_id else "stream"
        scope_key = self._build_feed_scope_key(url=url, camera_id=camera_id)
        feed_tracker = self._get_feed_tracker(scope_key)
        async for frame_idx, plates_raw, faces_raw, vehicles_raw in detect_stream_frames(
            url, frame_step, should_continue=should_continue
        ):
            best_vehicle = max(vehicles_raw, key=lambda v: v.confidence) if vehicles_raw else None
            plates_out = [self._enrich_plate(p, best_vehicle) for p in plates_raw]
            plates_out = await self._apply_ocr_postprocessing(plates_out, None)
            faces_out = await self._enrich_faces(faces_raw)
            vehicles_out = [self._map_vehicle(v) for v in vehicles_raw]
            feed_plate_events: list[dict] = []
            for plate in plates_out:
                tracker_plate = self._out_to_tracker_plate(plate)
                committed_feed = feed_tracker.observe(frame_idx, tracker_plate) or []
                for winner in committed_feed:
                    feed_plate_events.append(
                        self._plate_out_to_dict(self._tracker_plate_to_out(winner))
                    )
                if not self._passes_pre_filters_out(plate, camera_region):
                    continue
                committed = self._tracker.observe(tracker_plate) or []
                for winner in committed:
                    await self._log_committed(winner, source, camera_id, camera_name)
            if settings.PERSIST_FACE_EVENTS:
                for face in faces_out:
                    await self._save_face_event(face, camera_id, camera_name)
            yield {
                "frameIndex": frame_idx,
                "plates": [self._plate_out_to_dict(p) for p in plates_out],
                "feedPlateEvents": feed_plate_events,
                "feedPlates": [
                    self._plate_out_to_dict(self._tracker_plate_to_out(p))
                    for p in feed_tracker.get_feed_state()
                ],
                "faces": [self._face_out_to_dict(f) for f in faces_out],
                "vehicles": [self._vehicle_out_to_dict(v) for v in vehicles_out],
                "gunDetected": False,
            }
        for winner in self._tracker.flush_all():
            await self._log_committed(winner, source, camera_id, camera_name)

    # ── Enrichment helpers ───────────────────────────────────────────────────

    def _enrich_plate(self, p: PlateResult, best_vehicle: Optional[VehicleResult]) -> PlateOut:
        bb = p.bounding_box
        return PlateOut(
            text=normalize_plate(p.text),
            confidence=p.confidence,
            quality=p.quality,
            bounding_box={"x": bb.x, "y": bb.y, "width": bb.width, "height": bb.height, "rotation": bb.rotation},
            thumbnail=p.thumbnail,
            char_confidences=p.char_confidences,
            vehicle_make=best_vehicle.make if best_vehicle else None,
            vehicle_model=best_vehicle.model if best_vehicle else None,
            vehicle_color=best_vehicle.color if best_vehicle else None,
            vehicle_thumbnail=best_vehicle.thumbnail if best_vehicle else None,
        )

    async def _apply_ocr_postprocessing(
        self,
        plates: list[PlateOut],
        image_bytes: Optional[bytes],
    ) -> list[PlateOut]:
        if not plates:
            return plates

        known_plates = await self._known_plate_texts() if settings.ALPR_ENABLE_OCR_CORRECTION else []
        watchlist_plates = (
            await self._watchlist_plate_texts()
            if settings.ALPR_ENABLE_WATCHLIST_MULTI_READ
            else []
        )

        processed = []
        for plate in plates:
            if settings.ALPR_ENABLE_WATCHLIST_MULTI_READ and image_bytes and self._near_watchlist(plate.text, watchlist_plates):
                plate = self._multi_read_plate(plate, image_bytes)
            if settings.ALPR_ENABLE_OCR_CORRECTION and plate.quality < settings.ALPR_OCR_CORRECTION_MAX_QUALITY:
                plate = self._correct_plate_text(plate, known_plates)
            if settings.ALPR_FLAG_LOW_CONFIDENCE_CHARS:
                flagged = low_confidence_characters(
                    plate.text,
                    plate.char_confidences,
                    settings.ALPR_LOW_CHAR_CONFIDENCE_THRESHOLD,
                )
                plate.low_confidence_chars = flagged
                plate.manual_review_required = bool(flagged)
            processed.append(plate)
        return processed

    async def _known_plate_texts(self) -> list[str]:
        known: set[str] = set()
        try:
            if hasattr(self._persons, "get_known_plate_texts"):
                known.update(await self._persons.get_known_plate_texts())
        except Exception as exc:
            logger.warning("Known person plate lookup failed: %s", exc)
        try:
            known.update(await self._watchlist_plate_texts())
        except Exception as exc:
            logger.warning("Known watchlist plate lookup failed: %s", exc)
        return sorted(known)

    async def _watchlist_plate_texts(self) -> list[str]:
        try:
            if hasattr(self._watchlist, "get_plate_texts"):
                return await self._watchlist.get_plate_texts(active_only=True)
        except Exception as exc:
            logger.warning("Watchlist plate lookup failed: %s", exc)
        return []

    def _near_watchlist(self, plate_text: str, watchlist_plates: list[str]) -> bool:
        return any(
            levenshtein(plate_text, watchlist_plate) <= settings.ALPR_WATCHLIST_NEAR_MATCH_DISTANCE
            for watchlist_plate in watchlist_plates
        )

    def _multi_read_plate(self, plate: PlateOut, image_bytes: bytes) -> PlateOut:
        readings = [OcrReading(plate.text, plate.confidence, plate.quality)]
        detector = get_plate_detector()
        passes = max(0, settings.ALPR_WATCHLIST_MULTI_READ_PASSES - 1)
        for _ in range(passes):
            try:
                candidates = detector.detect(image_bytes, generate_thumbnail=False)
            except Exception as exc:
                logger.warning("Watchlist multi-read OCR pass failed: %s", exc)
                break
            nearest = self._nearest_candidate(plate, candidates)
            if nearest:
                readings.append(OcrReading(nearest.text, nearest.confidence, nearest.quality))
        voted = vote_ocr_readings(readings)
        if voted:
            plate.text = normalize_plate(voted.text)
            plate.confidence = voted.confidence
            plate.quality = voted.quality
        return plate

    def _nearest_candidate(self, plate: PlateOut, candidates: list[PlateResult]) -> Optional[PlateResult]:
        if not candidates:
            return None
        px = plate.bounding_box.get("x", 0) + plate.bounding_box.get("width", 0) / 2
        py = plate.bounding_box.get("y", 0) + plate.bounding_box.get("height", 0) / 2
        return min(
            candidates,
            key=lambda c: (
                abs((c.bounding_box.x + c.bounding_box.width / 2) - px)
                + abs((c.bounding_box.y + c.bounding_box.height / 2) - py)
            ),
        )

    def _correct_plate_text(self, plate: PlateOut, known_plates: list[str]) -> PlateOut:
        match, distance = best_fuzzy_match(
            plate.text,
            known_plates,
            max_distance=settings.ALPR_OCR_CORRECTION_MAX_DISTANCE,
        )
        if not match or match == plate.text:
            return plate
        plate.original_text = plate.text
        plate.text = match
        plate.corrected = True
        plate.correction_distance = distance
        plate.correction_source = "known_plate_db"
        return plate

    async def _enrich_faces(self, faces_raw: list[FaceResult]) -> list[FaceOut]:
        out = []
        for f in faces_raw:
            bb = f.bounding_box
            out.append(FaceOut(
                confidence=f.confidence,
                quality=f.quality,
                bounding_box={"x": bb.x, "y": bb.y, "width": bb.width, "height": bb.height, "rotation": 0},
                thumbnail=f.thumbnail,
                person_id=f.person_id,
                person_name=f.person_name,
                similarity=f.similarity,
                spoof_score=f.spoof_score,
                spoof_detected=f.spoof_detected,
                occluded=f.occluded,
            ))
        return out

    def _map_vehicle(self, v: VehicleResult) -> VehicleOut:
        bb = v.bounding_box
        return VehicleOut(
            make=v.make, model=v.model, color=v.color, type=v.type,
            confidence=v.confidence,
            bounding_box={"x": bb.x, "y": bb.y, "width": bb.width, "height": bb.height, "rotation": 0},
            thumbnail=v.thumbnail,
        )

    def _passes_pre_filters_out(self, p: PlateOut, region: Optional[str] = None) -> bool:
        from ml.plate_detector import PlateResult, BoundingBox, passes_pre_filters
        bb_dict = p.bounding_box
        pr = PlateResult(
            text=p.text,
            confidence=p.confidence,
            quality=p.quality,
            bounding_box=BoundingBox(**bb_dict),
        )
        return passes_pre_filters(pr, region)

    # ── Logging / alerts ─────────────────────────────────────────────────────

    async def _log_committed(self, tp: TrackerPlate, source: str,
                              camera_id=None, camera_name=None):
        now = time.time()
        cooldown_key = f"{tp.text}:{camera_id or ''}"
        last = self._recently_logged.get(cooldown_key)
        if last and (now - last) < PLATE_COOLDOWN_S:
            return
        self._recently_logged[cooldown_key] = now
        if len(self._recently_logged) > 500:
            cutoff = now - PLATE_COOLDOWN_S
            self._recently_logged = {k: v for k, v in self._recently_logged.items() if v > cutoff}
        plate_out = self._tracker_plate_to_out(tp)
        await self._log_and_alert(plate_out, source, camera_id, camera_name)

    async def _log_and_alert(self, plate: PlateOut, source: str,
                              camera_id=None, camera_name=None):
        event = await self._events.create({
            "plateText": plate.text,
            "confidence": plate.confidence,
            "source": source,
            "personId": plate.person_id,
            "personName": plate.person_name,
            "thumbnailBase64": plate.thumbnail,
            "x": plate.bounding_box.get("x", 0),
            "y": plate.bounding_box.get("y", 0),
            "width": plate.bounding_box.get("width", 0),
            "height": plate.bounding_box.get("height", 0),
            "vehicleMake": plate.vehicle_make,
            "vehicleModel": plate.vehicle_model,
            "vehicleColor": plate.vehicle_color,
            "vehicleThumbnail": plate.vehicle_thumbnail,
            "direction": plate.direction,
            "cameraId": camera_id,
            "cameraName": camera_name,
            "gunDetected": False,
        })
        notifications.emit_detection(event)
        await self._watchlist.check_and_alert(plate.text, event["id"], plate.thumbnail)
        if camera_id:
            await self._journeys.record_sighting({
                "plateText": plate.text,
                "cameraId": camera_id,
                "cameraName": camera_name,
                "thumbnailBase64": plate.thumbnail,
                "confidence": plate.confidence,
                "detectionEventId": event["id"],
            })

    async def _save_face_event(self, face: FaceOut, camera_id=None, camera_name=None):
        try:
            saved = await self._face_events.create({
                "personId": face.person_id,
                "personName": face.person_name,
                "confidence": face.confidence,
                "quality": face.quality,
                "spoofScore": face.spoof_score,
                "spoofDetected": face.spoof_detected,
                "occluded": face.occluded,
                "thumbnailBase64": face.thumbnail,
                "cameraId": camera_id,
                "cameraName": camera_name,
                "x": face.bounding_box.get("x", 0),
                "y": face.bounding_box.get("y", 0),
                "width": face.bounding_box.get("width", 0),
                "height": face.bounding_box.get("height", 0),
            })
            notifications.emit_face_event(saved)
        except Exception as exc:
            logger.warning("FaceEvent save failed: %s", exc)

    # ── Conversion helpers ────────────────────────────────────────────────────

    def _out_to_tracker_plate(self, p: PlateOut) -> TrackerPlate:
        bb = p.bounding_box
        return TrackerPlate(
            text=p.text, confidence=p.confidence, quality=p.quality,
            bounding_box=TrackerBBox(x=bb["x"], y=bb["y"], width=bb["width"], height=bb["height"]),
            thumbnail=p.thumbnail,
            low_confidence_chars=p.low_confidence_chars,
            manual_review_required=p.manual_review_required,
            original_text=p.original_text,
            corrected=p.corrected,
            correction_distance=p.correction_distance,
            correction_source=p.correction_source,
            direction=p.direction,
            vehicle_make=p.vehicle_make, vehicle_model=p.vehicle_model,
            vehicle_color=p.vehicle_color, person_id=p.person_id, person_name=p.person_name,
        )

    def _tracker_plate_to_out(self, tp: TrackerPlate) -> PlateOut:
        bb = tp.bounding_box
        return PlateOut(
            text=tp.text, confidence=tp.confidence, quality=tp.quality,
            bounding_box={"x": bb.x, "y": bb.y, "width": bb.width, "height": bb.height, "rotation": 0},
            thumbnail=tp.thumbnail,
            low_confidence_chars=tp.low_confidence_chars,
            manual_review_required=tp.manual_review_required,
            original_text=tp.original_text,
            corrected=tp.corrected,
            correction_distance=tp.correction_distance,
            correction_source=tp.correction_source,
            direction=tp.direction,
            vehicle_make=tp.vehicle_make, vehicle_model=tp.vehicle_model,
            vehicle_color=tp.vehicle_color, person_id=tp.person_id, person_name=tp.person_name,
        )

    @staticmethod
    def _plate_out_to_dict(p: PlateOut) -> dict:
        return {
            "text": p.text, "confidence": p.confidence, "quality": p.quality,
            "boundingBox": p.bounding_box, "thumbnail": p.thumbnail,
            "lowConfidenceChars": p.low_confidence_chars,
            "manualReviewRequired": p.manual_review_required,
            "originalText": p.original_text,
            "corrected": p.corrected,
            "correctionDistance": p.correction_distance,
            "correctionSource": p.correction_source,
            "direction": p.direction, "vehicleMake": p.vehicle_make,
            "vehicleModel": p.vehicle_model, "vehicleColor": p.vehicle_color,
            "vehicleThumbnail": p.vehicle_thumbnail, "personId": p.person_id,
            "personName": p.person_name,
        }

    @staticmethod
    def _face_out_to_dict(f: FaceOut) -> dict:
        return {
            "confidence": f.confidence, "quality": f.quality,
            "boundingBox": f.bounding_box, "thumbnail": f.thumbnail,
            "personId": f.person_id, "personName": f.person_name,
            "similarity": f.similarity, "spoofScore": f.spoof_score,
            "spoofDetected": f.spoof_detected, "occluded": f.occluded,
        }

    @staticmethod
    def _vehicle_out_to_dict(v: VehicleOut) -> dict:
        return {
            "make": v.make, "model": v.model, "color": v.color, "type": v.type,
            "confidence": v.confidence, "boundingBox": v.bounding_box,
            "thumbnail": v.thumbnail,
        }

    def _get_feed_tracker(self, scope_key: str) -> PlateFeedDeduplicator:
        tracker = self._feed_trackers.get(scope_key)
        if tracker is None:
            tracker = PlateFeedDeduplicator()
            self._feed_trackers[scope_key] = tracker
        return tracker

    @staticmethod
    def _build_feed_scope_key(url: str, camera_id: Optional[str] = None) -> str:
        if camera_id:
            return f"camera:{camera_id}"
        url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return f"stream:{url_hash}"
