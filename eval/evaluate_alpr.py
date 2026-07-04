
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.plate_detector import BoundingBox, PlateDetector, normalize_plate, passes_pre_filters

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_IOU_THRESHOLD = 0.5


def _bbox_from_any(value: Any) -> BoundingBox:
    if isinstance(value, dict):
        if {"x", "y", "width", "height"}.issubset(value):
            return BoundingBox(
                x=float(value["x"]),
                y=float(value["y"]),
                width=float(value["width"]),
                height=float(value["height"]),
            )
        if {"x1", "y1", "x2", "y2"}.issubset(value):
            return BoundingBox(
                x=float(value["x1"]),
                y=float(value["y1"]),
                width=float(value["x2"] - value["x1"]),
                height=float(value["y2"] - value["y1"]),
            )
    if isinstance(value, list) and len(value) == 4:
        x, y, w, h = value
        return BoundingBox(x=float(x), y=float(y), width=float(w), height=float(h))
    raise ValueError(f"Unsupported bbox format: {value!r}")


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx2, by2 = b.x + b.width, b.y + b.height
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def _levenshtein(a: str, b: str) -> int:
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


def _create_template(dataset: Path, gt_path: Path) -> None:
    images = sorted(
        p.relative_to(dataset).as_posix()
        for p in dataset.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    entries = [
        {
            "image": image,
            "plates": [
                {"bbox": {"x": 0, "y": 0, "width": 0, "height": 0}, "text": "ABC123"}
            ],
        }
        for image in images[:5]
    ]
    if not entries:
        entries = [{"image": "example.jpg", "plates": [{"bbox": [0, 0, 0, 0], "text": "ABC123"}]}]
    gt_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _load_ground_truth(gt_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(gt_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("images", payload.get("annotations", []))
    if not isinstance(payload, list):
        raise ValueError("Ground truth must be a list or a dict containing an images/annotations list")
    return payload


def _match_predictions(gt_plates, predictions, iou_threshold: float):
    pairs = []
    candidates = []
    for gt_idx, gt in enumerate(gt_plates):
        for pred_idx, pred in enumerate(predictions):
            iou = _iou(gt["bbox"], pred.bounding_box)
            if iou >= iou_threshold:
                candidates.append((iou, gt_idx, pred_idx))
    used_gt = set()
    used_pred = set()
    for iou, gt_idx, pred_idx in sorted(candidates, reverse=True):
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        pairs.append((gt_plates[gt_idx], predictions[pred_idx], iou))
    return pairs, used_gt, used_pred


def evaluate(dataset: Path, gt_path: Path, region: str | None, iou_threshold: float) -> dict[str, Any]:
    detector = PlateDetector()
    rows = _load_ground_truth(gt_path)
    totals = {
        "images": 0,
        "ground_truth": 0,
        "predictions": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "ocr_matches": 0,
        "ocr_samples": 0,
        "edit_distance": 0,
        "gt_chars": 0,
    }
    per_image = []

    for row in rows:
        image_rel = row["image"]
        image_path = dataset / image_rel
        gt_plates = [
            {"bbox": _bbox_from_any(plate["bbox"]), "text": normalize_plate(plate.get("text", ""))}
            for plate in row.get("plates", [])
        ]
        predictions = []
        if image_path.exists():
            predictions = [
                plate
                for plate in detector.detect(image_path.read_bytes(), generate_thumbnail=False)
                if passes_pre_filters(plate, region)
            ]

        pairs, used_gt, used_pred = _match_predictions(gt_plates, predictions, iou_threshold)
        tp = len(pairs)
        fp = len(predictions) - len(used_pred)
        fn = len(gt_plates) - len(used_gt)

        for gt, pred, _ in pairs:
            gt_text = gt["text"]
            pred_text = normalize_plate(pred.text)
            totals["ocr_samples"] += 1
            totals["ocr_matches"] += int(gt_text == pred_text)
            totals["edit_distance"] += _levenshtein(gt_text, pred_text)
            totals["gt_chars"] += len(gt_text)

        totals["images"] += 1
        totals["ground_truth"] += len(gt_plates)
        totals["predictions"] += len(predictions)
        totals["true_positive"] += tp
        totals["false_positive"] += fp
        totals["false_negative"] += fn
        per_image.append({
            "image": image_rel,
            "ground_truth": len(gt_plates),
            "predictions": len(predictions),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "missing": not image_path.exists(),
        })

    precision = _safe_div(totals["true_positive"], totals["true_positive"] + totals["false_positive"])
    recall = _safe_div(totals["true_positive"], totals["true_positive"] + totals["false_negative"])
    f1 = _safe_div(2 * precision * recall, precision + recall)
    exact = _safe_div(totals["ocr_matches"], totals["ocr_samples"])
    char_accuracy = 1.0 - _safe_div(totals["edit_distance"], totals["gt_chars"])

    return {
        "iou_threshold": iou_threshold,
        "region": region or "default",
        "detection": {"precision": precision, "recall": recall, "f1": f1},
        "ocr": {"exact_match_accuracy": exact, "char_level_accuracy": max(0.0, char_accuracy)},
        "counts": totals,
        "per_image": per_image,
    }


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _print_table(report: dict[str, Any]) -> None:
    rows = [
        ("Detection precision", report["detection"]["precision"]),
        ("Detection recall", report["detection"]["recall"]),
        ("Detection F1", report["detection"]["f1"]),
        ("OCR exact match", report["ocr"]["exact_match_accuracy"]),
        ("OCR char accuracy", report["ocr"]["char_level_accuracy"]),
    ]
    print("+---------------------+----------+")
    print("| Metric              | Value    |")
    print("+---------------------+----------+")
    for label, value in rows:
        print(f"| {label:<19} | {value:>8.3f} |")
    print("+---------------------+----------+")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ALPR detection and OCR metrics.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    gt_path = (args.ground_truth or dataset / "ground_truth.json").resolve()
    if not gt_path.exists():
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        _create_template(dataset, gt_path)
        print(f"Ground-truth file not found. Created sample template: {gt_path}")
        return 1

    report = evaluate(dataset, gt_path, args.region, args.iou)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_table(report)
    print(f"JSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
