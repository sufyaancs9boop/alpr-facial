from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PREDICTIONS_PATH = Path(__file__).resolve().with_name("predictions.json")
DEFAULT_GROUND_TRUTH_PATH = Path(__file__).resolve().with_name("ground_truth.json")
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().with_name("comparison_report.json")
DEFAULT_IOU_THRESHOLD = 0.5


@dataclass(frozen=True)
class BBox:
    """Axis-aligned rectangle used for IoU matching."""

    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height


@dataclass
class ComparisonStats:
    """Aggregate counters for the offline evaluation run."""

    images_processed: int = 0
    ground_truth_plates: int = 0
    predicted_plates: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    ocr_matches: int = 0
    ocr_samples: int = 0
    edit_distance: int = 0
    gt_chars: int = 0
    empty_ground_truth_images: int = 0
    empty_prediction_images: int = 0
    invalid_annotations_skipped: int = 0
    extra_prediction_images: int = 0


def load_json_list(path: Path, *, label: str) -> list[dict[str, Any]]:
    """Load a JSON file that must contain a top-level list of image records."""
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{label} file must contain a list of image records")
    return payload


def validate_unique_images(entries: Iterable[dict[str, Any]], *, label: str) -> None:
    """Ensure each image filename appears at most once in the given list."""
    seen: set[str] = set()
    duplicates: set[str] = set()

    for entry in entries:
        image_name = entry.get("image")
        if isinstance(image_name, str):
            if image_name in seen:
                duplicates.add(image_name)
            seen.add(image_name)

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate image names found in {label}: {duplicate_list}")


def _bbox_from_any(value: Any, *, image_name: str, source: str) -> BBox | None:
    """Coerce a bbox payload into a rectangle or return None if invalid."""
    if not isinstance(value, dict):
        warnings.warn(f"Skipping malformed annotation in {image_name} from {source}: bbox must be an object")
        return None

    required_keys = {"x", "y", "width", "height"}
    if not required_keys.issubset(value):
        warnings.warn(f"Skipping malformed annotation in {image_name} from {source}: bbox missing required fields")
        return None

    try:
        x = float(value["x"])
        y = float(value["y"])
        width = float(value["width"])
        height = float(value["height"])
    except (TypeError, ValueError):
        warnings.warn(f"Skipping malformed annotation in {image_name} from {source}: bbox values must be numeric")
        return None

    if not all(math.isfinite(v) for v in (x, y, width, height)):
        warnings.warn(f"Skipping malformed annotation in {image_name} from {source}: bbox values must be finite")
        return None
    if width <= 0 or height <= 0:
        warnings.warn(f"Skipping malformed annotation in {image_name} from {source}: width and height must be positive")
        return None
    if x < 0 or y < 0:
        warnings.warn(f"Skipping malformed annotation in {image_name} from {source}: x and y must be non-negative")
        return None

    return BBox(x=x, y=y, width=width, height=height)


def _exact_text(value: Any, *, image_name: str, source: str) -> str | None:
    """Validate that the OCR text field exists and is a string."""
    if not isinstance(value, str):
        warnings.warn(f"Skipping malformed annotation in {image_name} from {source}: missing text field")
        return None
    return value


def _levenshtein(a: str, b: str) -> int:
    """Compute the character edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def _iou(a: BBox, b: BBox) -> float:
    """Return intersection-over-union for two axis-aligned boxes."""
    intersection_x1 = max(a.x, b.x)
    intersection_y1 = max(a.y, b.y)
    intersection_x2 = min(a.x2, b.x2)
    intersection_y2 = min(a.y2, b.y2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height
    union_area = (a.width * a.height) + (b.width * b.height) - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _normalize_ground_truth_row(row: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]], int]:
    """Normalize one ground-truth image row and count skipped malformed annotations."""
    image_name = row.get("image")
    if not isinstance(image_name, str) or not image_name.strip():
        warnings.warn("Skipping malformed ground truth row: missing or invalid image name")
        return None, [], 1

    plates = row.get("plates", [])
    if not isinstance(plates, list):
        warnings.warn(f"Skipping malformed ground truth row for {image_name}: plates must be a list")
        return image_name, [], 1

    normalized: list[dict[str, Any]] = []
    skipped = 0
    for plate in plates:
        if not isinstance(plate, dict):
            warnings.warn(f"Skipping malformed ground truth annotation in {image_name}: expected object")
            skipped += 1
            continue

        bbox = _bbox_from_any(plate.get("bbox"), image_name=image_name, source="ground_truth")
        text = _exact_text(plate.get("text"), image_name=image_name, source="ground_truth")
        if bbox is None or text is None:
            skipped += 1
            continue

        normalized.append({"bbox": bbox, "text": text})

    return image_name, normalized, skipped


def _normalize_prediction_row(row: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]], int]:
    """Normalize one prediction image row and count skipped malformed annotations."""
    image_name = row.get("image")
    if not isinstance(image_name, str) or not image_name.strip():
        warnings.warn("Skipping malformed prediction row: missing or invalid image name")
        return None, [], 1

    predictions = row.get("predictions", [])
    if not isinstance(predictions, list):
        warnings.warn(f"Skipping malformed prediction row for {image_name}: predictions must be a list")
        return image_name, [], 1

    normalized: list[dict[str, Any]] = []
    skipped = 0
    for prediction in predictions:
        if not isinstance(prediction, dict):
            warnings.warn(f"Skipping malformed prediction in {image_name}: expected object")
            skipped += 1
            continue

        bbox = _bbox_from_any(prediction.get("bbox"), image_name=image_name, source="predictions")
        text = _exact_text(prediction.get("text"), image_name=image_name, source="predictions")
        if bbox is None or text is None:
            skipped += 1
            continue

        normalized.append({"bbox": bbox, "text": text})

    return image_name, normalized, skipped


def _match_predictions(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], float]], set[int], set[int]]:
    """Greedily match predictions to ground-truth boxes by highest IoU."""
    candidate_pairs: list[tuple[float, int, int]] = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            score = _iou(gt["bbox"], pred["bbox"])
            if score >= iou_threshold:
                candidate_pairs.append((score, gt_index, pred_index))

    matched_ground_truth: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[tuple[dict[str, Any], dict[str, Any], float]] = []

    for score, gt_index, pred_index in sorted(candidate_pairs, reverse=True):
        if gt_index in matched_ground_truth or pred_index in matched_predictions:
            continue
        matched_ground_truth.add(gt_index)
        matched_predictions.add(pred_index)
        matches.append((ground_truth[gt_index], predictions[pred_index], score))

    return matches, matched_ground_truth, matched_predictions


def compare_predictions(
    predictions_path: Path,
    ground_truth_path: Path,
    iou_threshold: float,
) -> tuple[dict[str, Any], ComparisonStats]:
    """Compare saved predictions to verified ground truth and return a full report."""
    stats = ComparisonStats()
    predictions_rows = load_json_list(predictions_path, label="predictions")
    ground_truth_rows = load_json_list(ground_truth_path, label="ground truth")

    validate_unique_images(predictions_rows, label="predictions")
    validate_unique_images(ground_truth_rows, label="ground truth")

    predictions_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in predictions_rows:
        image_name, predictions, skipped = _normalize_prediction_row(row)
        stats.invalid_annotations_skipped += skipped
        if image_name is None:
            continue
        predictions_by_image[image_name] = predictions

    ground_truth_by_image: dict[str, list[dict[str, Any]]] = {}
    ordered_images: list[str] = []
    for row in ground_truth_rows:
        image_name, plates, skipped = _normalize_ground_truth_row(row)
        stats.invalid_annotations_skipped += skipped
        if image_name is None:
            continue
        ground_truth_by_image[image_name] = plates
        ordered_images.append(image_name)

    extra_prediction_images = sorted(set(predictions_by_image) - set(ground_truth_by_image))
    if extra_prediction_images:
        warnings.warn(
            "Ignoring predictions for images not present in ground truth: "
            + ", ".join(extra_prediction_images)
        )

    stats.extra_prediction_images = len(extra_prediction_images)
    per_image: list[dict[str, Any]] = []

    for image_name in ordered_images:
        gt_plates = ground_truth_by_image.get(image_name, [])
        predicted_plates = predictions_by_image.get(image_name, [])

        stats.images_processed += 1
        stats.ground_truth_plates += len(gt_plates)
        stats.predicted_plates += len(predicted_plates)

        if not gt_plates:
            stats.empty_ground_truth_images += 1
        if not predicted_plates:
            stats.empty_prediction_images += 1

        matches, matched_gt, matched_pred = _match_predictions(gt_plates, predicted_plates, iou_threshold)
        true_positive = len(matches)
        false_positive = len(predicted_plates) - len(matched_pred)
        false_negative = len(gt_plates) - len(matched_gt)

        stats.true_positive += true_positive
        stats.false_positive += false_positive
        stats.false_negative += false_negative

        for gt_plate, pred_plate, _ in matches:
            gt_text = gt_plate["text"]
            pred_text = pred_plate["text"]
            stats.ocr_samples += 1
            stats.ocr_matches += int(gt_text == pred_text)
            stats.edit_distance += _levenshtein(gt_text, pred_text)
            stats.gt_chars += len(gt_text)

        per_image.append(
            {
                "image": image_name,
                "ground_truth": len(gt_plates),
                "predictions": len(predicted_plates),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "missing_prediction": image_name not in predictions_by_image,
            }
        )

    precision = _safe_div(stats.true_positive, stats.true_positive + stats.false_positive)
    recall = _safe_div(stats.true_positive, stats.true_positive + stats.false_negative)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    exact_match_accuracy = _safe_div(stats.ocr_matches, stats.ocr_samples)
    char_level_accuracy = 1.0 - _safe_div(stats.edit_distance, stats.gt_chars)

    report = {
        "iou_threshold": iou_threshold,
        "detection": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "ocr": {
            "exact_match_accuracy": exact_match_accuracy,
            "char_level_accuracy": max(0.0, char_level_accuracy),
        },
        "counts": {
            "images": stats.images_processed,
            "ground_truth_plates": stats.ground_truth_plates,
            "predicted_plates": stats.predicted_plates,
            "true_positive": stats.true_positive,
            "false_positive": stats.false_positive,
            "false_negative": stats.false_negative,
            "ocr_matches": stats.ocr_matches,
            "ocr_samples": stats.ocr_samples,
            "edit_distance": stats.edit_distance,
            "gt_chars": stats.gt_chars,
            "empty_ground_truth_images": stats.empty_ground_truth_images,
            "empty_prediction_images": stats.empty_prediction_images,
            "extra_prediction_images": stats.extra_prediction_images,
            "invalid_annotations_skipped": stats.invalid_annotations_skipped,
        },
        "per_image": per_image,
    }
    return report, stats


def _print_report(report: dict[str, Any], stats: ComparisonStats, output_path: Path) -> None:
    """Print a concise evaluation summary for the user."""
    print(f"Images processed: {stats.images_processed}")
    print(f"Ground truth plates: {stats.ground_truth_plates}")
    print(f"Predicted plates: {stats.predicted_plates}")
    print(f"Empty ground truth images: {stats.empty_ground_truth_images}")
    print(f"Empty prediction images: {stats.empty_prediction_images}")
    print(f"Invalid annotations skipped: {stats.invalid_annotations_skipped}")
    print(f"Extra prediction images ignored: {stats.extra_prediction_images}")
    print(f"Detection precision: {report['detection']['precision']:.3f}")
    print(f"Detection recall: {report['detection']['recall']:.3f}")
    print(f"Detection F1: {report['detection']['f1']:.3f}")
    print(f"OCR exact match accuracy: {report['ocr']['exact_match_accuracy']:.3f}")
    print(f"OCR char-level accuracy: {report['ocr']['char_level_accuracy']:.3f}")
    print(f"Output file location: {output_path}")


def main() -> int:
    """CLI entry point for comparing model predictions against verified ground truth."""
    parser = argparse.ArgumentParser(
        description="Compare predictions.json against ground_truth.json and compute accuracy metrics."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path to predictions.json.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_PATH,
        help="Path to ground_truth.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the comparison report JSON.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="IoU threshold used to match prediction boxes to ground-truth boxes.",
    )
    args = parser.parse_args()

    predictions_path = args.predictions.resolve()
    ground_truth_path = args.ground_truth.resolve()
    output_path = args.output.resolve()

    try:
        report, stats = compare_predictions(predictions_path, ground_truth_path, args.iou)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_report(report, stats, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())