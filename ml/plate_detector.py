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

# Pakistani plate regex (same as plate.util.ts)
_PLATE_RE = re.compile(r"^[A-Z]{2,4}\d{3,8}$")

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


def is_valid_pakistani_plate(normalized: str) -> bool:
    return bool(_PLATE_RE.match(normalized))


def passes_pre_filters(result: PlateResult) -> bool:
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
    if not is_valid_pakistani_plate(normalized):
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

        return results
