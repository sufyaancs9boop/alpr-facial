"""
License plate detection + OCR using fast-alpr (MIT).
Wraps fast-alpr's ONNX pipeline (YOLOv9-t detection + CCT global OCR model).
Applies the same pre-filters as the old ROC-based system.
"""
import re
import base64
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Regional plate patterns. Pakistan is the default set; other regions can be
# added here without changing filter call sites.
PLATE_PATTERNS = {
    "PAKISTAN": [
        re.compile(r"^[A-Z]{2,4}\d{3,8}$"),
        re.compile(r"^[A-Z]{2,4}[-\s]?\d{3,8}$"),
        re.compile(r"^[A-Z]{2,3}[-\s]?\d{2}[-\s]?\d{3,4}$"),
        re.compile(r"^[A-Z]{2,3}[-\s]?\d{3,4}[-\s]?[A-Z]$"),
        re.compile(r"^[A-Z]{3}[-\s]?\d{1,4}$"),
        re.compile(r"^\d{2}[-\s]?[A-Z]{1,3}[-\s]?\d{3,4}$"),
    ],
}

REGION_ALIASES = {
    "PK": "PAKISTAN",
    "PAK": "PAKISTAN",
    "PAKISTANI": "PAKISTAN",
    "NORTH_AMERICAN": "PAKISTAN",
}

# Pre-filter thresholds (mirrored from alpr.service.ts)
MIN_PLATE_PX_WIDTH = 40
MIN_CONFIDENCE = 0.70
MIN_ASPECT_RATIO = 1.1


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0


@dataclass
class PlateResult:
    text: str
    confidence: float
    quality: float
    bounding_box: BoundingBox
    thumbnail: Optional[str] = None  # base64 JPEG data URI


def normalize_plate(text: str) -> str:
    return re.sub(r"[\s\-_]", "", text).upper()


def normalize_region(region: Optional[str] = None) -> str:
    if not region:
        try:
            from config import settings
            region = settings.DEFAULT_PLATE_REGION
        except Exception:
            region = "PAKISTAN"
    key = re.sub(r"[\s\-]+", "_", str(region).strip().upper())
    return REGION_ALIASES.get(key, key)


def is_valid_plate(text: str, region: Optional[str] = None) -> bool:
    region_key = normalize_region(region)
    patterns = PLATE_PATTERNS.get(region_key) or PLATE_PATTERNS["PAKISTAN"]
    raw = (text or "").strip().upper()
    normalized = normalize_plate(raw)
    return any(pattern.match(raw) or pattern.match(normalized) for pattern in patterns)


def is_valid_pakistani_plate(text: str) -> bool:
    return is_valid_plate(text, "PAKISTAN")


def _intersection_over_union(a: BoundingBox, b: BoundingBox) -> float:
    ax2 = a.x + a.width
    ay2 = a.y + a.height
    bx2 = b.x + b.width
    by2 = b.y + b.height
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance_ratio(a: BoundingBox, b: BoundingBox) -> float:
    acx = a.x + a.width / 2
    acy = a.y + a.height / 2
    bcx = b.x + b.width / 2
    bcy = b.y + b.height / 2
    distance = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    scale = max(a.width, a.height, b.width, b.height, 1.0)
    return distance / scale


def deduplicate_plates(
    plates: list[PlateResult],
    iou_threshold: Optional[float] = None,
    center_distance_ratio: Optional[float] = None,
) -> list[PlateResult]:
    if iou_threshold is None or center_distance_ratio is None:
        from config import settings
        if iou_threshold is None:
            iou_threshold = settings.ALPR_DEDUP_IOU_THRESHOLD
        if center_distance_ratio is None:
            center_distance_ratio = settings.ALPR_DEDUP_CENTER_DISTANCE_RATIO

    winners: list[PlateResult] = []
    for plate in sorted(plates, key=lambda p: p.confidence, reverse=True):
        overlaps = any(
            _intersection_over_union(plate.bounding_box, kept.bounding_box) >= iou_threshold
            or _center_distance_ratio(plate.bounding_box, kept.bounding_box) <= center_distance_ratio
            for kept in winners
        )
        if not overlaps:
            winners.append(plate)
    return winners


def passes_pre_filters(result: PlateResult, region: Optional[str] = None) -> bool:
    bb = result.bounding_box
    w, h = bb.width, max(bb.height, 1)
    ratio = w / h

    if w < MIN_PLATE_PX_WIDTH:
        logger.debug("PREFILTER SKIP [too narrow] '%s' w=%.0fpx", result.text, w)
        return False
    if result.confidence < MIN_CONFIDENCE:
        logger.debug("PREFILTER SKIP [low conf] '%s' conf=%.0f%%", result.text, result.confidence * 100)
        return False
    if ratio < MIN_ASPECT_RATIO:
        logger.debug("PREFILTER SKIP [portrait] '%s' w/h=%.2f", result.text, ratio)
        return False
    normalized = normalize_plate(result.text)
    if not is_valid_plate(result.text, region):
        logger.debug("PREFILTER SKIP [regex] '%s' → '%s'", result.text, normalized)
        return False
    logger.debug("PREFILTER PASS '%s' w=%.0fpx conf=%.0f%%", normalized, w, result.confidence * 100)
    return True


def _crop_thumbnail(image_np: np.ndarray, bb: BoundingBox) -> Optional[str]:
    """Crop bounding box from image and return as base64 JPEG data URI."""
    try:
        h_img, w_img = image_np.shape[:2]
        x1 = max(0, int(bb.x))
        y1 = max(0, int(bb.y))
        x2 = min(w_img, int(bb.x + bb.width))
        y2 = min(h_img, int(bb.y + bb.height))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image_np[y1:y2, x1:x2]
        pil_img = Image.fromarray(crop[..., ::-1])  # BGR → RGB
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return b64
    except Exception:
        return None


class PlateDetector:
    """
    Lazy-initialized fast-alpr detector.
    Call detect(image_bytes) to get a list of PlateResult.
    """

    def __init__(self):
        self._alpr = None

    def _ensure_loaded(self):
        if self._alpr is not None:
            return
        try:
            from fast_alpr import ALPR
            # Global model handles Latin-character plates (incl. Pakistani)
            self._alpr = ALPR(
                detector_model="yolo-v9-t-384-license-plate-end2end",
                ocr_model="global-plates-mobile-vit-v2-model",
            )
            logger.info("fast-alpr loaded (YOLOv9-t + global OCR model)")
        except ImportError as exc:
            raise RuntimeError("fast-alpr not installed. Run: pip install fast-alpr[onnx-cpu]") from exc

    def detect(self, image_bytes: bytes, generate_thumbnail: bool = True) -> list[PlateResult]:
        self._ensure_loaded()

        import cv2
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image_np = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image_np is None:
            logger.warning("Could not decode image for plate detection")
            return []

        try:
            raw_results = self._alpr.predict(image_np)
        except Exception as exc:
            logger.error("fast-alpr inference error: %s", exc)
            return []

        results: list[PlateResult] = []
        for r in raw_results:
            if r.detection is None:
                continue
            # BoundingBox has .x1 .y1 .x2 .y2 attributes (pixel coords)
            bb_raw = r.detection.bounding_box
            x1, y1, x2, y2 = bb_raw.x1, bb_raw.y1, bb_raw.x2, bb_raw.y2
            bb = BoundingBox(x=float(x1), y=float(y1),
                             width=float(x2 - x1), height=float(y2 - y1))

            ocr_text = (r.ocr.text if r.ocr else "") or ""
            # ocr.confidence is a per-character list; take the mean as overall quality
            if r.ocr and r.ocr.confidence:
                conf_vals = r.ocr.confidence
                ocr_quality = float(sum(conf_vals) / len(conf_vals)) if isinstance(conf_vals, list) else float(conf_vals)
            else:
                ocr_quality = 0.0

            text = normalize_plate(ocr_text)
            plate = PlateResult(
                text=text,
                confidence=float(r.detection.confidence),
                quality=ocr_quality,
                bounding_box=bb,
                thumbnail=_crop_thumbnail(image_np, bb) if generate_thumbnail else None,
            )
            results.append(plate)

        # TODO: Add temporal dedup here if a frame-to-frame tracker becomes part
        # of the detector layer. Current temporal grouping lives in services.
        return deduplicate_plates(results)
