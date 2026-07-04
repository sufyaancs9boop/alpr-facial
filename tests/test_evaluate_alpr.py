import importlib.util
import json
import shutil
import uuid
from pathlib import Path

from ml.plate_detector import BoundingBox, PlateResult


def _load_eval_module():
    path = Path(__file__).resolve().parents[1] / "eval" / "evaluate_alpr.py"
    spec = importlib.util.spec_from_file_location("evaluate_alpr", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeDetector:
    def detect(self, image_bytes, generate_thumbnail=False):
        return [
            PlateResult(
                text="ABC123",
                confidence=0.95,
                quality=0.95,
                bounding_box=BoundingBox(x=10, y=10, width=100, height=30),
            ),
            PlateResult(
                text="LEA1235",
                confidence=0.90,
                quality=0.90,
                bounding_box=BoundingBox(x=200, y=10, width=100, height=30),
            ),
            PlateResult(
                text="ICT221234",
                confidence=0.85,
                quality=0.85,
                bounding_box=BoundingBox(x=400, y=10, width=100, height=30),
            ),
        ]


def test_evaluate_alpr_detection_and_ocr_metrics():
    module = _load_eval_module()
    module.PlateDetector = _FakeDetector
    tmp_dir = Path(__file__).resolve().parent / f".tmp_eval_{uuid.uuid4().hex}"
    try:
        tmp_dir.mkdir()
        image_path = tmp_dir / "frame.jpg"
        image_path.write_bytes(b"fake-image")
        gt_path = tmp_dir / "ground_truth.json"
        gt_path.write_text(json.dumps([
            {
                "image": "frame.jpg",
                "plates": [
                    {"bbox": {"x": 10, "y": 10, "width": 100, "height": 30}, "text": "ABC123"},
                    {"bbox": {"x": 200, "y": 10, "width": 100, "height": 30}, "text": "LEA1234"},
                ],
            }
        ]))

        report = module.evaluate(tmp_dir, gt_path, region="PAKISTAN", iou_threshold=0.5)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    assert report["detection"]["precision"] == 2 / 3
    assert report["detection"]["recall"] == 1.0
    assert report["detection"]["f1"] == 0.8
    assert report["ocr"]["exact_match_accuracy"] == 0.5
    assert report["ocr"]["char_level_accuracy"] == 12 / 13
