from services.plate_tracker import PlateFeedDeduplicator, _PlateInput, _BBox


def _plate(text: str, confidence: float = 0.9, x: float = 10.0, y: float = 10.0) -> _PlateInput:
    return _PlateInput(
        text=text,
        confidence=confidence,
        quality=confidence,
        bounding_box=_BBox(x=x, y=y, width=100.0, height=30.0),
        thumbnail=None,
    )


def test_same_plate_across_frames_emits_once():
    dedup = PlateFeedDeduplicator(
        window_frames=5,
        min_observations=2,
        max_edit_distance=2,
        cooldown_frames=30,
        spatial_distance_ratio=0.35,
    )

    assert dedup.observe_many(0, [_plate("ABC123")]) == []
    emitted = dedup.observe_many(1, [_plate("ABC123", confidence=0.95)])
    assert len(emitted) == 1
    assert emitted[0].text == "ABC123"
    assert dedup.observe_many(2, [_plate("ABC123")]) == []


def test_ocr_variation_merges_single_committed_plate():
    dedup = PlateFeedDeduplicator(
        window_frames=5,
        min_observations=2,
        max_edit_distance=2,
        cooldown_frames=30,
        spatial_distance_ratio=0.35,
    )

    assert dedup.observe_many(0, [_plate("AYL963", confidence=0.90)]) == []
    emitted = dedup.observe_many(1, [_plate("AYL967", confidence=0.85)])
    assert len(emitted) == 1
    assert emitted[0].text in {"AYL963", "AYL967"}
    assert dedup.observe_many(2, [_plate("AYL963", confidence=0.92)]) == []


def test_different_plates_do_not_merge():
    dedup = PlateFeedDeduplicator(
        window_frames=5,
        min_observations=2,
        max_edit_distance=1,
        cooldown_frames=30,
        spatial_distance_ratio=0.35,
    )

    assert dedup.observe_many(0, [_plate("ABC123", x=10)]) == []
    assert dedup.observe_many(1, [_plate("LEA1234", x=300)]) == []


def test_same_plate_can_emit_again_after_cooldown():
    dedup = PlateFeedDeduplicator(
        window_frames=5,
        min_observations=2,
        max_edit_distance=2,
        cooldown_frames=3,
        spatial_distance_ratio=0.35,
    )

    assert dedup.observe_many(0, [_plate("ABC123")]) == []
    first = dedup.observe_many(1, [_plate("ABC123")])
    assert len(first) == 1
    assert first[0].text == "ABC123"

    assert dedup.observe_many(2, [_plate("ABC123")]) == []
    assert dedup.observe_many(3, [_plate("ABC123")]) == []
    assert dedup.observe_many(4, [_plate("ABC123")]) == []
    second = dedup.observe_many(5, [_plate("ABC123")])
    assert len(second) == 1
    assert second[0].text == "ABC123"


def test_moving_plate_still_matches_on_text_not_spatial():
    dedup = PlateFeedDeduplicator(
        window_frames=5,
        min_observations=2,
        max_edit_distance=2,
        cooldown_frames=30,
        spatial_distance_ratio=0.35,
    )

    assert dedup.observe_many(0, [_plate("ABC123", x=10)]) == []
    emitted = dedup.observe_many(1, [_plate("ABC123", x=500)])
    assert len(emitted) == 1
    assert emitted[0].text == "ABC123"
    assert dedup.observe_many(2, [_plate("ABC123", x=900)]) == []


def test_feed_state_accumulates_confirmed_plates():
    dedup = PlateFeedDeduplicator(
        window_frames=5,
        min_observations=2,
        max_edit_distance=2,
        cooldown_frames=30,
        spatial_distance_ratio=0.35,
    )

    assert dedup.observe_many(0, [_plate("AVT889")]) == []
    emitted_avt = dedup.observe_many(1, [_plate("AVT889")])
    assert len(emitted_avt) == 1
    assert emitted_avt[0].text == "AVT889"
    assert [p.text for p in dedup.get_feed_state()] == ["AVT889"]

    assert dedup.observe_many(2, [_plate("AYL967")]) == []
    emitted_ayl = dedup.observe_many(3, [_plate("AYL967")])
    assert len(emitted_ayl) == 1
    assert emitted_ayl[0].text == "AYL967"
    assert [p.text for p in dedup.get_feed_state()] == ["AVT889", "AYL967"]

    assert dedup.observe_many(10, []) == []
    assert [p.text for p in dedup.get_feed_state()] == ["AVT889", "AYL967"]


def test_different_tracker_scopes_do_not_merge():
    tracker_a = PlateFeedDeduplicator(
        window_frames=5,
        min_observations=2,
        max_edit_distance=2,
        cooldown_frames=30,
        spatial_distance_ratio=0.35,
    )
    tracker_b = PlateFeedDeduplicator(
        window_frames=5,
        min_observations=2,
        max_edit_distance=2,
        cooldown_frames=30,
        spatial_distance_ratio=0.35,
    )

    assert tracker_a.observe_many(0, [_plate("ABC123")]) == []
    assert tracker_b.observe_many(0, [_plate("ABC123")]) == []

    emitted_a = tracker_a.observe_many(1, [_plate("ABC123")])
    emitted_b = tracker_b.observe_many(1, [_plate("ABC123")])
    assert len(emitted_a) == 1
    assert len(emitted_b) == 1
