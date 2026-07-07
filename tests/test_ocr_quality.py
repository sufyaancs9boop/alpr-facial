import asyncio

import pytest

from ml.ocr_quality import (
    OcrReading,
    best_fuzzy_match,
    calibration_report,
    character_error_rate,
    low_confidence_characters,
    vote_ocr_readings,
    word_error_rate,
)


def test_cer_wer_and_calibration_report():
    assert character_error_rate("ABC123", "ABC128") == 1 / 6
    assert word_error_rate("ABC123 LEA1234", "ABC123 LEA1235") == 1 / 2

    report = calibration_report([
        {"confidence": 0.95, "correct": True, "cer": 0.0},
        {"confidence": 0.91, "correct": False, "cer": 0.2},
        {"confidence": 0.82, "correct": True, "cer": 0.0},
    ])

    assert report[0] == {"bucket": "0.9-1.0", "samples": 2, "accuracy": 0.5, "avg_cer": 0.1}
    assert report[1] == {"bucket": "0.8-0.9", "samples": 1, "accuracy": 1.0, "avg_cer": 0.0}


def test_fuzzy_match_requires_unique_close_plate():
    assert best_fuzzy_match("ABC124", ["ABC123", "LEA1234"], max_distance=1) == ("ABC123", 1)
    assert best_fuzzy_match("ABC124", ["ABC123", "ABC125"], max_distance=1) == (None, 1)
    assert best_fuzzy_match("ABC999", ["ABC123"], max_distance=1) == (None, None)


def test_vote_ocr_readings_uses_majority_then_quality():
    winner = vote_ocr_readings([
        OcrReading("ABC123", 0.80, 0.80),
        OcrReading("ABC128", 0.99, 0.99),
        OcrReading("ABC123", 0.82, 0.90),
    ])

    assert winner.text == "ABC123"
    assert winner.confidence == 0.81
    assert winner.quality == 0.8500000000000001


def test_low_confidence_character_flags():
    assert low_confidence_characters("ABC123", [0.9, 0.6, 0.7, 0.2], 0.7) == [
        {"index": 1, "char": "B", "confidence": 0.6},
        {"index": 3, "char": "1", "confidence": 0.2},
    ]


def test_alpr_postprocessing_corrects_and_flags(monkeypatch):
    pytest.importorskip("cv2")
    from services.alpr_service import AlprService, PlateOut
    from config import settings

    monkeypatch.setattr(settings, "ALPR_ENABLE_OCR_CORRECTION", True)
    monkeypatch.setattr(settings, "ALPR_OCR_CORRECTION_MAX_QUALITY", 0.85)
    monkeypatch.setattr(settings, "ALPR_OCR_CORRECTION_MAX_DISTANCE", 1)
    monkeypatch.setattr(settings, "ALPR_FLAG_LOW_CONFIDENCE_CHARS", True)
    monkeypatch.setattr(settings, "ALPR_LOW_CHAR_CONFIDENCE_THRESHOLD", 0.7)
    monkeypatch.setattr(settings, "ALPR_ENABLE_WATCHLIST_MULTI_READ", False)

    class _Persons:
        async def get_known_plate_texts(self):
            return ["ABC123"]

    class _Watchlist:
        async def get_plate_texts(self, active_only=True):
            return []

    service = AlprService(None, _Persons(), None, _Watchlist(), None, None)
    plate = PlateOut(
        text="ABC128",
        confidence=0.95,
        quality=0.80,
        bounding_box={"x": 0, "y": 0, "width": 100, "height": 30},
        char_confidences=[0.95, 0.95, 0.95, 0.95, 0.95, 0.4],
    )

    processed = asyncio.run(service._apply_ocr_postprocessing([plate], None))[0]

    assert processed.text == "ABC123"
    assert processed.original_text == "ABC128"
    assert processed.corrected is True
    assert processed.correction_distance == 1
    assert processed.manual_review_required is True
    assert processed.low_confidence_chars == [{"index": 5, "char": "3", "confidence": 0.4}]
