"""
Port of vehicle-tracker.ts — spatial-only tracker for video session mode.
Commits the single best reading per vehicle (largest bounding box = closest = clearest).
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class _BBox:
    x: float
    y: float
    width: float
    height: float


@dataclass
class _PlateInput:
    text: str
    confidence: float
    quality: float
    bounding_box: _BBox
    thumbnail: Optional[str] = None
    direction: Optional[str] = None
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_color: Optional[str] = None
    person_id: Optional[str] = None
    person_name: Optional[str] = None


@dataclass
class _Reading:
    plate: _PlateInput
    area: float


@dataclass
class _Track:
    id: str
    readings: list[_Reading] = field(default_factory=list)
    last_centroid_x: float = 0
    last_centroid_y: float = 0
    last_width: float = 0
    last_height: float = 0
    last_seen: float = field(default_factory=time.time)
    first_centroid_x: float = 0


def _cx(p: _PlateInput) -> float:
    return p.bounding_box.x + p.bounding_box.width / 2


def _cy(p: _PlateInput) -> float:
    return p.bounding_box.y + p.bounding_box.height / 2


def _is_nearby(px: float, py: float, pw: float, ph: float, track: _Track) -> bool:
    ref_w = max(pw, track.last_width)
    ref_h = max(ph, track.last_height)
    return abs(px - track.last_centroid_x) < ref_w * 4 and abs(py - track.last_centroid_y) < ref_h * 4


def _direction(first_x: float, last_x: float) -> str:
    d = last_x - first_x
    if abs(d) < 40:
        return "stationary"
    return "right" if d > 0 else "left"


class VehicleTracker:
    """Spatial-only tracker for video sessions."""

    def __init__(self, idle_ms: int = 8_000, min_readings: int = 1):
        self._tracks: dict[str, _Track] = {}
        self._idle_s = idle_ms / 1000
        self._min_readings = min_readings

    def observe(self, plate: _PlateInput) -> list[_PlateInput]:
        now = time.time()
        committed = self._flush_expired(now)
        px, py = _cx(plate), _cy(plate)
        area = plate.bounding_box.width * plate.bounding_box.height

        track: Optional[_Track] = None
        for t in self._tracks.values():
            if _is_nearby(px, py, plate.bounding_box.width, plate.bounding_box.height, t):
                track = t
                break

        if track is None:
            track = _Track(
                id=str(uuid.uuid4()),
                last_centroid_x=px,
                last_centroid_y=py,
                last_width=plate.bounding_box.width,
                last_height=plate.bounding_box.height,
                last_seen=now,
                first_centroid_x=px,
            )
            self._tracks[track.id] = track
        else:
            track.last_centroid_x = px
            track.last_centroid_y = py
            track.last_width = plate.bounding_box.width
            track.last_height = plate.bounding_box.height

        track.readings.append(_Reading(plate=plate, area=area))
        track.last_seen = now
        return committed

    def flush_all(self) -> list[_PlateInput]:
        results = []
        for track_id, track in list(self._tracks.items()):
            if len(track.readings) >= self._min_readings:
                results.append(self._pick_best(track))
            del self._tracks[track_id]
        return results

    def _flush_expired(self, now: float) -> list[_PlateInput]:
        results = []
        for track_id, track in list(self._tracks.items()):
            if now - track.last_seen >= self._idle_s:
                if len(track.readings) >= self._min_readings:
                    results.append(self._pick_best(track))
                del self._tracks[track_id]
        return results

    def _pick_best(self, track: _Track) -> _PlateInput:
        from collections import Counter
        # Vote on the most common text reading across all frames
        votes = Counter(r.plate.text for r in track.readings)
        top_text, _ = votes.most_common(1)[0]
        # Among readings that share the winning text, pick highest OCR quality then area
        candidates = [r for r in track.readings if r.plate.text == top_text]
        best = max(candidates, key=lambda r: (r.plate.quality, r.area))
        plate = _PlateInput(**vars(best.plate))
        plate.direction = _direction(track.first_centroid_x, track.last_centroid_x)
        return plate
