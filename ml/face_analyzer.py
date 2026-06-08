"""
Face detection + recognition using InsightFace (ArcFace).
Replaces ROC's roc_represent_face_ex + gallery search.
"""
import base64
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.4  # cosine similarity; tune against enrolled faces


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0


@dataclass
class FaceResult:
    confidence: float
    quality: float
    bounding_box: BoundingBox
    embedding: Optional[list[float]] = None
    thumbnail: Optional[str] = None
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    similarity: Optional[float] = None
    spoof_score: Optional[float] = None
    spoof_detected: bool = False
    occluded: bool = False


def _crop_thumbnail(image_np: np.ndarray, bb: BoundingBox) -> Optional[str]:
    try:
        import cv2
        h_img, w_img = image_np.shape[:2]
        pad = int(max(bb.width, bb.height) * 0.15)
        x1 = max(0, int(bb.x) - pad)
        y1 = max(0, int(bb.y) - pad)
        x2 = min(w_img, int(bb.x + bb.width) + pad)
        y2 = min(h_img, int(bb.y + bb.height) + pad)
        crop = image_np[y1:y2, x1:x2]
        pil_img = Image.fromarray(crop[..., ::-1])
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return b64
    except Exception:
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))


class FaceAnalyzer:
    """
    Lazy-initialized InsightFace FaceAnalysis.
    Handles detection, embedding, and gallery search.
    """

    def __init__(self, models_dir: str = "data/models"):
        self._app = None
        self._models_dir = models_dir
        # Gallery: person_id → list of embeddings (np.ndarray shape [512])
        self._gallery: dict[str, list[np.ndarray]] = {}
        self._person_names: dict[str, str] = {}

    def _ensure_loaded(self):
        if self._app is not None:
            return
        try:
            import insightface
            from insightface.app import FaceAnalysis
            os.makedirs(self._models_dir, exist_ok=True)
            self._app = FaceAnalysis(
                name="buffalo_s",
                root=self._models_dir,
                providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace buffalo_s loaded")
        except ImportError as exc:
            raise RuntimeError("insightface not installed. Run: pip install insightface") from exc

    # ── Gallery management ──────────────────────────────────────────────────

    def load_gallery(self, gallery: dict[str, dict]):
        """Load gallery from Person records. gallery = {person_id: {name, embeddings: [[...]]}}"""
        self._gallery.clear()
        self._person_names.clear()
        for pid, data in gallery.items():
            self._person_names[pid] = data.get("name", "")
            embeddings = [np.array(e, dtype=np.float32) for e in data.get("embeddings", [])]
            if embeddings:
                self._gallery[pid] = embeddings

    def add_person(self, person_id: str, name: str, embeddings: list[list[float]]):
        self._person_names[person_id] = name
        self._gallery[person_id] = [np.array(e, dtype=np.float32) for e in embeddings]

    def remove_person(self, person_id: str):
        self._gallery.pop(person_id, None)
        self._person_names.pop(person_id, None)

    def extract_embedding(self, image_bytes: bytes) -> Optional[list[float]]:
        """Extract face embedding from a single-face image (for enrollment)."""
        embedding, _ = self.extract_embedding_and_thumbnail(image_bytes)
        return embedding

    def extract_embedding_and_thumbnail(self, image_bytes: bytes) -> tuple[Optional[list[float]], Optional[str]]:
        """Extract embedding + cropped face thumbnail from an enrollment image."""
        self._ensure_loaded()
        import cv2
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        faces = self._app.get(img)
        if not faces:
            return None, None
        best = max(faces, key=lambda f: f.det_score)
        x1, y1, x2, y2 = best.bbox
        bb = BoundingBox(x=float(x1), y=float(y1), width=float(x2 - x1), height=float(y2 - y1))
        thumbnail = _crop_thumbnail(img, bb)
        return best.embedding.tolist(), thumbnail

    def search_gallery(self, embedding: np.ndarray) -> tuple[Optional[str], Optional[str], float]:
        """Return (person_id, person_name, similarity) of best match, or (None, None, 0)."""
        best_pid, best_name, best_sim = None, None, 0.0
        for pid, embs in self._gallery.items():
            for emb in embs:
                sim = _cosine_similarity(embedding, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_pid = pid
                    best_name = self._person_names.get(pid)
        if best_sim >= SIMILARITY_THRESHOLD:
            return best_pid, best_name, best_sim
        return None, None, 0.0

    # ── Detection ───────────────────────────────────────────────────────────

    def detect(self, image_bytes: bytes, generate_thumbnail: bool = True) -> list[FaceResult]:
        self._ensure_loaded()
        import cv2
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []

        try:
            faces = self._app.get(img)
        except Exception as exc:
            logger.error("InsightFace inference error: %s", exc)
            return []

        results: list[FaceResult] = []
        for face in faces:
            bbox_arr = face.bbox  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox_arr
            bb = BoundingBox(
                x=float(x1), y=float(y1),
                width=float(x2 - x1), height=float(y2 - y1),
            )
            embedding = face.embedding  # np.ndarray [512]
            pid, pname, sim = self.search_gallery(embedding)

            result = FaceResult(
                confidence=float(face.det_score),
                quality=float(face.det_score),  # InsightFace doesn't expose a separate quality score
                bounding_box=bb,
                embedding=embedding.tolist(),
                thumbnail=_crop_thumbnail(img, bb) if generate_thumbnail else None,
                person_id=pid,
                person_name=pname,
                similarity=sim if pid else None,
            )
            results.append(result)

        return results
