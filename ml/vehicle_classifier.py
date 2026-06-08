"""
Vehicle detection (YOLOv8n) + color classification (HSV histogram).
Replaces ROC's roc_represent_object_ex.
"""
import base64
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
import io
import cv2

logger = logging.getLogger(__name__)

# YOLOv8 COCO class IDs for vehicles
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# 12-bucket HSV color map (matches ROC output color names)
_COLOR_RANGES = [
    ("red",     0,   10),
    ("orange", 11,   25),
    ("yellow", 26,   35),
    ("green",  36,   85),
    ("cyan",   86,  100),
    ("blue",  101,  130),
    ("purple",131,  160),
    ("pink",  161,  170),
    ("red",   171,  180),  # red wraps
]


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0


@dataclass
class VehicleResult:
    make: Optional[str]
    model: Optional[str]
    color: Optional[str]
    type: Optional[str]
    confidence: float
    bounding_box: BoundingBox
    thumbnail: Optional[str] = None


def _dominant_color(image_np: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[str]:
    """Compute dominant color of vehicle crop using HSV histogram."""
    try:
        crop = image_np[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Mask out low-saturation (grey/white/black) pixels
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = (sat > 50) & (val > 50)
        hue_vals = hsv[:, :, 0][mask]

        if len(hue_vals) < 50:
            # Mostly achromatic — pick white/grey/black by brightness
            mean_val = float(np.mean(val))
            if mean_val > 200:
                return "white"
            elif mean_val > 100:
                return "silver"
            else:
                return "black"

        # Bucket hues
        counts: dict[str, int] = {}
        for h in hue_vals:
            for color, lo, hi in _COLOR_RANGES:
                if lo <= h <= hi:
                    counts[color] = counts.get(color, 0) + 1
                    break
        return max(counts, key=counts.get) if counts else None
    except Exception:
        return None


def _crop_thumbnail(image_np: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[str]:
    try:
        crop = image_np[y1:y2, x1:x2]
        pil_img = Image.fromarray(crop[..., ::-1])
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return b64
    except Exception:
        return None


class VehicleClassifier:
    """Lazy-initialized YOLOv8n vehicle detector + HSV color classifier."""

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")  # auto-downloaded on first run
            logger.info("YOLOv8n vehicle detector loaded")
        except ImportError as exc:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics") from exc

    def detect(self, image_bytes: bytes, generate_thumbnail: bool = True) -> list[VehicleResult]:
        self._ensure_loaded()

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image_np = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image_np is None:
            return []

        try:
            results = self._model(image_np, verbose=False)[0]
        except Exception as exc:
            logger.error("YOLOv8 inference error: %s", exc)
            return []

        vehicles: list[VehicleResult] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            vehicle_type = VEHICLE_CLASS_IDS[cls_id]
            color = _dominant_color(image_np, x1, y1, x2, y2)
            bb = BoundingBox(
                x=float(x1), y=float(y1),
                width=float(x2 - x1), height=float(y2 - y1),
            )
            vehicles.append(VehicleResult(
                make=None,   # make/model not available without a commercial model
                model=None,
                color=color,
                type=vehicle_type,
                confidence=conf,
                bounding_box=bb,
                thumbnail=_crop_thumbnail(image_np, x1, y1, x2, y2) if generate_thumbnail else None,
            ))

        return vehicles
