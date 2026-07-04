from ml.plate_detector import BoundingBox, PlateResult, deduplicate_plates


def _plate(text, confidence, x, y, width=100, height=30):
    return PlateResult(
        text=text,
        confidence=confidence,
        quality=confidence,
        bounding_box=BoundingBox(x=x, y=y, width=width, height=height),
    )


def test_deduplicate_plates_keeps_highest_confidence_overlap():
    plates = [
        _plate("ABC123", 0.72, 10, 10),
        _plate("ABC123", 0.95, 12, 11),
        _plate("LEA1234", 0.80, 250, 10),
    ]

    deduped = deduplicate_plates(plates, iou_threshold=0.5, center_distance_ratio=0.25)

    assert [p.text for p in deduped] == ["ABC123", "LEA1234"]
    assert deduped[0].confidence == 0.95


def test_deduplicate_plates_keeps_separate_boxes():
    plates = [
        _plate("ABC123", 0.95, 10, 10),
        _plate("LEA1234", 0.90, 180, 10),
    ]

    deduped = deduplicate_plates(plates, iou_threshold=0.5, center_distance_ratio=0.25)

    assert len(deduped) == 2
