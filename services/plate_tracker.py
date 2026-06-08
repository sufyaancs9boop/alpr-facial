"""
Port of plate-tracker.ts — groups multi-frame observations into sessions.
Commits best reading per session after minObservations or idleMs timeout.
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 3:
        return 99
    n = len(b)
    dp = list(range(n + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            tmp = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


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
    # vehicle enrichment
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_color: Optional[str] = None
    person_id: Optional[str] = None
    person_name: Optional[str] = None


@dataclass
class _VoteEntry:
    count: int = 0
    best: Optional[_PlateInput] = None


@dataclass
class _Session:
    anchor_text: str
    votes: dict = field(default_factory=dict)  # text → _VoteEntry
    total_votes: int = 0
    last_seen: float = field(default_factory=time.time)
    first_centroid_x: float = 0
    last_centroid_x: float = 0
    last_centroid_y: float = 0
    last_width: float = 0
    last_height: float = 0


def _centroid_x(p: _PlateInput) -> float:
    return p.bounding_box.x + p.bounding_box.width / 2


def _centroid_y(p: _PlateInput) -> float:
    return p.bounding_box.y + p.bounding_box.height / 2


def _spatially_close(plate: _PlateInput, session: _Session) -> bool:
    cx = _centroid_x(plate)
    cy = _centroid_y(plate)
    ref_w = max(plate.bounding_box.width, session.last_width)
    ref_h = max(plate.bounding_box.height, session.last_height)
    dx = abs(cx - session.last_centroid_x)
    dy = abs(cy - session.last_centroid_y)
    return dx < ref_w * 4 and dy < ref_h * 4


def _compute_direction(first_x: float, last_x: float) -> str:
    delta = last_x - first_x
    if abs(delta) < 40:
        return "stationary"
    return "right" if delta > 0 else "left"


def _pick_winner(session: _Session) -> _PlateInput:
    winner_entry: Optional[_VoteEntry] = None
    for entry in session.votes.values():
        if (winner_entry is None
                or entry.count > winner_entry.count
                or (entry.count == winner_entry.count and entry.best.confidence > winner_entry.best.confidence)):
            winner_entry = entry
    plate = _PlateInput(**vars(winner_entry.best))
    plate.direction = _compute_direction(session.first_centroid_x, session.last_centroid_x)
    return plate


class PlateTracker:
    """Thread-unsafe but asyncio-safe (single-threaded event loop)."""

    def __init__(self, commit_after_ms: int = 8_000, max_edit_distance: int = 2, min_observations: int = 1):
        self._sessions: dict[str, _Session] = {}
        self._commit_after_s = commit_after_ms / 1000
        self._max_edit = max_edit_distance
        self._min_obs = min_observations

    def observe(self, plate: _PlateInput) -> list[_PlateInput]:
        now = time.time()
        committed = self._flush_expired(now)
        cx = _centroid_x(plate)
        cy = _centroid_y(plate)

        matched: Optional[_Session] = None
        matched_key: Optional[str] = None
        for k, session in self._sessions.items():
            if (_levenshtein(plate.text, session.anchor_text) <= self._max_edit
                    or _spatially_close(plate, session)):
                matched = session
                matched_key = k
                break

        if matched is None:
            key = f"{plate.text}_{now}"
            matched = _Session(
                anchor_text=plate.text,
                last_seen=now,
                first_centroid_x=cx,
                last_centroid_x=cx,
                last_centroid_y=cy,
                last_width=plate.bounding_box.width,
                last_height=plate.bounding_box.height,
            )
            self._sessions[key] = matched
            matched_key = key
        else:
            matched.last_centroid_x = cx
            matched.last_centroid_y = cy
            matched.last_width = plate.bounding_box.width
            matched.last_height = plate.bounding_box.height

        entry = matched.votes.get(plate.text)
        if entry:
            entry.count += 1
            if plate.confidence > entry.best.confidence:
                entry.best = plate
        else:
            matched.votes[plate.text] = _VoteEntry(count=1, best=plate)
        matched.total_votes += 1
        matched.last_seen = now

        if matched.total_votes >= self._min_obs:
            winner = _pick_winner(matched)
            committed.append(winner)
            del self._sessions[matched_key]

        return committed

    def flush_all(self) -> list[_PlateInput]:
        results = []
        for key, session in list(self._sessions.items()):
            if session.total_votes >= self._min_obs:
                results.append(_pick_winner(session))
            del self._sessions[key]
        return results

    def _flush_expired(self, now: float) -> list[_PlateInput]:
        results = []
        for key, session in list(self._sessions.items()):
            if now - session.last_seen >= self._commit_after_s:
                if session.total_votes >= self._min_obs:
                    results.append(_pick_winner(session))
                del self._sessions[key]
        return results
