from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def character_error_rate(expected: str, predicted: str) -> float:
    return _safe_div(levenshtein(expected, predicted), len(expected))


def word_error_rate(expected: str, predicted: str) -> float:
    expected_words = expected.split()
    predicted_words = predicted.split()
    if not expected_words:
        return 0.0 if not predicted_words else 1.0
    return _safe_div(_sequence_edit_distance(expected_words, predicted_words), len(expected_words))


def low_confidence_characters(
    text: str,
    confidences: Optional[list[float]],
    threshold: float,
) -> list[dict]:
    if not confidences:
        return []
    flagged = []
    for idx, confidence in enumerate(confidences):
        if confidence < threshold:
            flagged.append({
                "index": idx,
                "char": text[idx] if idx < len(text) else "",
                "confidence": float(confidence),
            })
    return flagged


def confidence_bucket(confidence: float, step: float = 0.1) -> str:
    confidence = max(0.0, min(1.0, float(confidence)))
    lower = int(confidence / step) * step
    if confidence >= 1.0:
        lower = 1.0 - step
    upper = min(1.0, lower + step)
    return f"{lower:.1f}-{upper:.1f}"


def calibration_report(samples: Iterable[dict], step: float = 0.1) -> list[dict]:
    buckets: dict[str, dict] = defaultdict(lambda: {"samples": 0, "correct": 0, "cer_total": 0.0})
    for sample in samples:
        bucket = confidence_bucket(float(sample.get("confidence", 0.0)), step)
        entry = buckets[bucket]
        entry["samples"] += 1
        entry["correct"] += int(bool(sample.get("correct", False)))
        entry["cer_total"] += float(sample.get("cer", 0.0))

    rows = []
    for bucket in sorted(buckets.keys(), reverse=True):
        entry = buckets[bucket]
        samples_count = entry["samples"]
        rows.append({
            "bucket": bucket,
            "samples": samples_count,
            "accuracy": _safe_div(entry["correct"], samples_count),
            "avg_cer": _safe_div(entry["cer_total"], samples_count),
        })
    return rows


def best_fuzzy_match(
    predicted: str,
    known_plates: Iterable[str],
    max_distance: int = 1,
) -> tuple[Optional[str], Optional[int]]:
    candidates = []
    for plate in known_plates:
        if not plate:
            continue
        distance = levenshtein(predicted, plate)
        if distance <= max_distance:
            candidates.append((distance, plate))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_distance = candidates[0][0]
    best = [plate for distance, plate in candidates if distance == best_distance]
    if len(best) != 1:
        return None, best_distance
    return best[0], best_distance


@dataclass
class OcrReading:
    text: str
    confidence: float
    quality: float


def vote_ocr_readings(readings: Iterable[OcrReading]) -> Optional[OcrReading]:
    items = list(readings)
    if not items:
        return None
    votes = Counter(item.text for item in items)
    top_count = votes.most_common(1)[0][1]
    top_texts = {text for text, count in votes.items() if count == top_count}
    candidates = [item for item in items if item.text in top_texts]
    best_text = max(
        top_texts,
        key=lambda text: (
            _safe_div(sum(item.quality for item in candidates if item.text == text), votes[text]),
            _safe_div(sum(item.confidence for item in candidates if item.text == text), votes[text]),
            text,
        ),
    )
    best_items = [item for item in items if item.text == best_text]
    return OcrReading(
        text=best_text,
        confidence=_safe_div(sum(item.confidence for item in best_items), len(best_items)),
        quality=_safe_div(sum(item.quality for item in best_items), len(best_items)),
    )


def _sequence_edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, 1):
        curr = [i]
        for j, token_b in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (token_a != token_b)))
        prev = curr
    return prev[-1]


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0
