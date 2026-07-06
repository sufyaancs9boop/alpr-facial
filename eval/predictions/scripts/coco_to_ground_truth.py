from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

DEFAULT_COCO_PATH = Path(__file__).resolve().with_name("instances_Validation.json")
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().with_name("ground_truth.json")
LICENSE_PLATE_CATEGORY_NAME = "LicensePlate"


@dataclass
class ConversionStats:
	"""Aggregate counters for the COCO-to-ground-truth conversion."""

	images_processed: int = 0
	plates_exported: int = 0
	empty_images: int = 0
	invalid_annotations_skipped: int = 0


@dataclass(frozen=True)
class GroundTruthPlate:
	"""Normalized plate record expected by evaluate.py."""

	x: float
	y: float
	width: float
	height: float
	text: str


def load_coco_annotations(coco_path: Path) -> dict[str, Any]:
	"""Load and validate a COCO instances export."""
	if not coco_path.exists():
		raise FileNotFoundError(f"COCO instances file not found: {coco_path}")

	try:
		payload = json.loads(coco_path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as exc:
		raise ValueError(f"Invalid JSON in COCO file: {exc}") from exc

	if not isinstance(payload, dict):
		raise ValueError("COCO export must be a JSON object")

	for key in ("images", "annotations", "categories"):
		if key not in payload or not isinstance(payload[key], list):
			raise ValueError(f"COCO export must contain a list field named {key!r}")

	return payload


def _coerce_number(value: Any, *, field_name: str, image_name: str) -> float | None:
	"""Parse a numeric value and reject non-finite inputs."""
	try:
		number = float(value)
	except (TypeError, ValueError):
		warnings.warn(f"Skipping malformed annotation in {image_name}: {field_name} is not numeric")
		return None

	if not math.isfinite(number):
		warnings.warn(f"Skipping malformed annotation in {image_name}: {field_name} is not finite")
		return None
	return number


def _json_number(value: float) -> int | float:
	"""Serialize whole numbers as integers while preserving decimals when needed."""
	return int(value) if float(value).is_integer() else value


def _normalize_text(value: Any, *, image_name: str) -> str | None:
	"""Validate the OCR text attribute and preserve it exactly."""
	if not isinstance(value, str):
		warnings.warn(f"Skipping malformed annotation in {image_name}: missing text attribute")
		return None
	return value


def _normalize_bbox(value: Any, *, image_name: str) -> GroundTruthPlate | None:
	"""Convert COCO [x, y, width, height] boxes into ground-truth records."""
	if not isinstance(value, list) or len(value) != 4:
		warnings.warn(f"Skipping malformed annotation in {image_name}: bbox must be a 4-item list")
		return None

	x = _coerce_number(value[0], field_name="bbox[0]", image_name=image_name)
	y = _coerce_number(value[1], field_name="bbox[1]", image_name=image_name)
	width = _coerce_number(value[2], field_name="bbox[2]", image_name=image_name)
	height = _coerce_number(value[3], field_name="bbox[3]", image_name=image_name)
	if None in (x, y, width, height):
		return None

	if x < 0 or y < 0 or width <= 0 or height <= 0:
		warnings.warn(f"Skipping malformed annotation in {image_name}: bbox coordinates must be non-negative and non-zero")
		return None

	return GroundTruthPlate(x=x, y=y, width=width, height=height, text="")


def _build_image_lookup(images: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
	"""Index images by COCO image id and ensure filenames are unique."""
	lookup: dict[int, dict[str, Any]] = {}
	seen_filenames: set[str] = set()

	for image in images:
		if not isinstance(image, dict):
			raise ValueError("Each COCO image entry must be an object")

		image_id = image.get("id")
		file_name = image.get("file_name")
		if not isinstance(image_id, int):
			raise ValueError("Each COCO image entry must have an integer id")
		if not isinstance(file_name, str) or not file_name.strip():
			raise ValueError(f"COCO image {image_id} is missing a valid file_name")
		if file_name in seen_filenames:
			raise ValueError(f"Duplicate image filename in COCO export: {file_name}")

		seen_filenames.add(file_name)
		lookup[image_id] = image

	return lookup


def _build_category_lookup(categories: list[dict[str, Any]]) -> dict[int, str]:
	"""Index category ids by name for label validation."""
	lookup: dict[int, str] = {}
	for category in categories:
		if not isinstance(category, dict):
			raise ValueError("Each COCO category entry must be an object")
		category_id = category.get("id")
		name = category.get("name")
		if not isinstance(category_id, int) or not isinstance(name, str):
			raise ValueError("Each COCO category entry must have integer id and string name")
		lookup[category_id] = name
	return lookup


def convert_coco_to_ground_truth(coco_path: Path) -> tuple[list[dict[str, Any]], ConversionStats]:
	"""Convert COCO instances annotations into evaluate.py-compatible ground truth JSON."""
	payload = load_coco_annotations(coco_path)
	images = payload["images"]
	annotations = payload["annotations"]
	categories = payload["categories"]

	image_lookup = _build_image_lookup(images)
	category_lookup = _build_category_lookup(categories)
	license_plate_category_ids = {category_id for category_id, name in category_lookup.items() if name == LICENSE_PLATE_CATEGORY_NAME}
	if not license_plate_category_ids:
		raise ValueError(f"COCO export does not contain category {LICENSE_PLATE_CATEGORY_NAME!r}")

	annotations_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_lookup}
	stats = ConversionStats(images_processed=len(image_lookup))

	for annotation in annotations:
		if not isinstance(annotation, dict):
			warnings.warn("Skipping malformed annotation: expected object")
			stats.invalid_annotations_skipped += 1
			continue

		image_id = annotation.get("image_id")
		category_id = annotation.get("category_id")
		bbox = annotation.get("bbox")
		attributes = annotation.get("attributes", {})

		if image_id not in image_lookup:
			warnings.warn(f"Skipping annotation with unknown image_id {image_id!r}")
			stats.invalid_annotations_skipped += 1
			continue
		if category_id not in license_plate_category_ids:
			continue
		if not isinstance(attributes, dict):
			warnings.warn(f"Skipping malformed annotation for image_id {image_id}: attributes must be an object")
			stats.invalid_annotations_skipped += 1
			continue

		plate = _normalize_bbox(bbox, image_name=image_lookup[image_id]["file_name"])
		if plate is None:
			stats.invalid_annotations_skipped += 1
			continue

		text = _normalize_text(attributes.get("text"), image_name=image_lookup[image_id]["file_name"])
		if text is None:
			stats.invalid_annotations_skipped += 1
			continue

		annotations_by_image[image_id].append(
			{
				"bbox": {
					"x": _json_number(plate.x),
					"y": _json_number(plate.y),
					"width": _json_number(plate.width),
					"height": _json_number(plate.height),
				},
				"text": text,
			}
		)
		stats.plates_exported += 1

	output: list[dict[str, Any]] = []
	for image in images:
		image_id = image["id"]
		file_name = image["file_name"]
		plates = annotations_by_image.get(image_id, [])
		if not plates:
			stats.empty_images += 1
		output.append({"image": file_name, "plates": plates})

	return output, stats


def write_ground_truth(data: list[dict[str, Any]], output_path: Path) -> None:
	"""Write the converted ground truth JSON to disk."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
	"""CLI entry point for converting COCO instances annotations to ground_truth.json."""
	parser = argparse.ArgumentParser(
		description="Convert COCO instances JSON into evaluate.py ground_truth.json."
	)
	parser.add_argument(
		"--coco",
		type=Path,
		default=DEFAULT_COCO_PATH,
		help="Path to the COCO instances export.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT_PATH,
		help="Path to write ground_truth.json.",
	)
	args = parser.parse_args()

	coco_path = args.coco.resolve()
	output_path = args.output.resolve()

	try:
		data, stats = convert_coco_to_ground_truth(coco_path)
		write_ground_truth(data, output_path)
	except Exception as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 1

	print(f"Images processed: {stats.images_processed}")
	print(f"Plates exported: {stats.plates_exported}")
	print(f"Empty images: {stats.empty_images}")
	print(f"Invalid annotations skipped: {stats.invalid_annotations_skipped}")
	print(f"Output file location: {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())