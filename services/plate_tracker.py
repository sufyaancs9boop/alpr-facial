"""
Port of plate-tracker.ts — groups multi-frame observations into sessions.
Commits best reading per session after minObservations or idleMs timeout.
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from config import settings
from ml.plate_detector import normalize_plate

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
    low_confidence_chars: Optional[list[dict]] = None
    manual_review_required: bool = False
    original_text: Optional[str] = None
    corrected: bool = False
    correction_distance: Optional[int] = None
    correction_source: Optional[str] = None
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


@dataclass
class _FeedTrack:
    track_id: str
    first_frame: int
    last_frame: int
    observations: int = 0
    text_votes: dict[str, int] = field(default_factory=dict)
    best_plate: Optional[_PlateInput] = None
    best_confidence: float = -1.0
    emitted: bool = False
    last_emitted_frame: Optional[int] = None
    first_centroid_x: float = 0.0
    last_centroid_x: float = 0.0
    last_centroid_y: float = 0.0
    last_width: float = 0.0
    last_height: float = 0.0


class PlateFeedDeduplicator:
    """
    Multi-frame feed deduplication for stream/video payloads.

    Uses processed-frame indices to group OCR readings into short-lived tracks,
    emitting only when stable and suppressing repeats by cooldown.
    """

    def __init__(
        self,
        window_frames: int = settings.ALPR_FEED_DEDUP_WINDOW_FRAMES,
        min_observations: int = settings.ALPR_FEED_MIN_OBSERVATIONS,
        max_edit_distance: int = settings.ALPR_FEED_MAX_EDIT_DISTANCE,
        cooldown_frames: int = settings.ALPR_FEED_COOLDOWN_FRAMES,
        spatial_distance_ratio: float = settings.ALPR_FEED_SPATIAL_DISTANCE_RATIO,
    ):
        self._window_frames = max(1, int(window_frames))
        self._min_observations = max(1, int(min_observations))
        self._max_edit_distance = max(0, int(max_edit_distance))
        self._cooldown_frames = max(0, int(cooldown_frames))
        self._spatial_distance_ratio = max(0.0, float(spatial_distance_ratio))
        self._tracks: dict[str, _FeedTrack] = {}
        self._last_emitted_by_text: dict[str, int] = {}
        self._feed_state: dict[str, _PlateInput] = {}
        self._next_track_id = 1

    def observe(self, frame_idx: int, plate: _PlateInput) -> list[_PlateInput]:
        return self.observe_many(frame_idx, [plate])

    def observe_many(self, frame_idx: int, plates: list[_PlateInput]) -> list[_PlateInput]:
        self._expire_tracks(frame_idx)
        emitted: list[_PlateInput] = []
        for plate in plates:
            normalized_text = normalize_plate(plate.text or "")
            if not normalized_text:
                continue
            plate.text = normalized_text
            track = self._match_track(frame_idx, plate)
            if track is None:
                track = self._create_track(frame_idx, plate)
            self._update_track(track, frame_idx, plate)
            maybe_emitted = self._commit_if_ready(track, frame_idx)
            if maybe_emitted is not None:
                key = normalize_plate(maybe_emitted.text)
                self._feed_state[key] = maybe_emitted
                emitted.append(maybe_emitted)
        return emitted

    def get_feed_state(self) -> list[_PlateInput]:
        return list(self._feed_state.values())

    def flush(self, frame_idx: Optional[int] = None) -> list[_PlateInput]:
        if frame_idx is not None:
            self._expire_tracks(frame_idx)
        return []

    def _expire_tracks(self, frame_idx: int) -> None:
        for key, track in list(self._tracks.items()):
            if frame_idx - track.last_frame > self._window_frames:
                del self._tracks[key]

    def _match_track(self, frame_idx: int, plate: _PlateInput) -> Optional[_FeedTrack]:
        candidates: list[tuple[_FeedTrack, float]] = []
        for track in self._tracks.values():
            if track.emitted:
                continue
            if frame_idx - track.last_frame > self._window_frames:
                continue
            winner_text = self._winning_text(track)
            if winner_text is None:
                continue
            edit_distance = _levenshtein(plate.text, winner_text)
            text_match = plate.text == winner_text or edit_distance <= self._max_edit_distance
            if not text_match:
                continue
            candidates.append((track, self._spatial_distance_ratio_to_track(plate, track)))

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][0]
        candidates.sort(key=lambda item: item[1])
        return candidates[0][0]

    def _create_track(self, frame_idx: int, plate: _PlateInput) -> _FeedTrack:
        track_id = f"feed-{self._next_track_id}"
        self._next_track_id += 1
        cx = _centroid_x(plate)
        cy = _centroid_y(plate)
        track = _FeedTrack(
            track_id=track_id,
            first_frame=frame_idx,
            last_frame=frame_idx,
            first_centroid_x=cx,
            last_centroid_x=cx,
            last_centroid_y=cy,
            last_width=plate.bounding_box.width,
            last_height=plate.bounding_box.height,
        )
        self._tracks[track_id] = track
        return track

    def _update_track(self, track: _FeedTrack, frame_idx: int, plate: _PlateInput) -> None:
        track.last_frame = frame_idx
        track.observations += 1
        track.text_votes[plate.text] = track.text_votes.get(plate.text, 0) + 1

        score = (plate.confidence, plate.quality)
        best_score = (
            track.best_plate.confidence,
            track.best_plate.quality,
        ) if track.best_plate is not None else (-1.0, -1.0)
        if track.best_plate is None or score > best_score:
            track.best_plate = _PlateInput(**vars(plate))
            track.best_confidence = plate.confidence

        track.last_centroid_x = _centroid_x(plate)
        track.last_centroid_y = _centroid_y(plate)
        track.last_width = plate.bounding_box.width
        track.last_height = plate.bounding_box.height

    def _commit_if_ready(self, track: _FeedTrack, frame_idx: int) -> Optional[_PlateInput]:
        if track.emitted:
            return None
        if track.observations < self._min_observations:
            return None

        winning_text = self._winning_text(track)
        if not winning_text:
            return None

        last_emitted = self._last_emitted_by_text.get(winning_text)
        if last_emitted is not None and (frame_idx - last_emitted) <= self._cooldown_frames:
            return None

        if track.best_plate is None:
            return None

        output = _PlateInput(**vars(track.best_plate))
        output.text = winning_text
        output.direction = _compute_direction(track.first_centroid_x, track.last_centroid_x)
        track.emitted = True
        track.last_emitted_frame = frame_idx
        self._last_emitted_by_text[winning_text] = frame_idx
        return output

    def _winning_text(self, track: _FeedTrack) -> Optional[str]:
        if not track.text_votes:
            return None
        winners = sorted(track.text_votes.items(), key=lambda kv: kv[1], reverse=True)
        top_count = winners[0][1]
        top_texts = [t for t, c in winners if c == top_count]
        if len(top_texts) == 1:
            return top_texts[0]
        if track.best_plate and track.best_plate.text in top_texts:
            return track.best_plate.text
        return top_texts[0]

    def _spatial_distance_ratio_to_track(self, plate: _PlateInput, track: _FeedTrack) -> float:
        cx = _centroid_x(plate)
        cy = _centroid_y(plate)
        dx = abs(cx - track.last_centroid_x)
        dy = abs(cy - track.last_centroid_y)
        ref = max(
            plate.bounding_box.width,
            plate.bounding_box.height,
            track.last_width,
            track.last_height,
            1.0,
        )
        return ((dx * dx + dy * dy) ** 0.5) / ref
