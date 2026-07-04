from ml.plate_detector import is_valid_plate, normalize_plate


def test_valid_pakistani_plate_variants():
    valid = [
        "ABC123",
        "LEA-1234",
        "ICT 22 1234",
        "LHR 1234 A",
        "BLA-9876",
        "22 LE 1234",
    ]

    for sample in valid:
        assert is_valid_plate(sample, "PAKISTAN"), sample


def test_invalid_pakistani_plate_samples():
    invalid = [
        "",
        "A123",
        "ABCDE123",
        "ABC12@",
        "123456",
        "LE-12",
    ]

    for sample in invalid:
        assert not is_valid_plate(sample, "PAKISTAN"), sample


def test_normalize_plate_removes_common_separators():
    assert normalize_plate(" lea-1234 ") == "LEA1234"
