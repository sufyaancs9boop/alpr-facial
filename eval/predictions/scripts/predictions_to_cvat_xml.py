from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

DEFAULT_PREDICTIONS_PATH = Path(__file__).resolve().with_name("predictions.json")
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().with_name("predictions_cvat.xml")
DEFAULT_IMAGES_DIR = ROOT / "eval" / "dataset" / "images-alpr"
LICENSE_PLATE_LABEL = "LicensePlate"
TEXT_ATTRIBUTE = "text"


@dataclass
class ValidationStats:
	total_images: int = 0
	total_predictions: int = 0
	images_without_detections: int = 0
	invalid_predictions_skipped: int = 0


@dataclass(frozen=True)
class NormalizedPrediction:
	x: float
	y: float
	width: float
	height: float
	text: str

	@property
	def xbr(self) -> float:
		return self.x + self.width

	@property
	def ybr(self) -> float:
		return self.y + self.height



def load_predictions(predictions_path: Path) -> list[dict[str, Any]]:
	"""Load and parse the backend predictions JSON payload."""
	if not predictions_path.exists():
		raise FileNotFoundError(f"predictions.json not found: {predictions_path}")

	payload = json.loads(predictions_path.read_text(encoding="utf-8"))
	if not isinstance(payload, list):
		raise ValueError("predictions.json must contain a list of image prediction entries")
	return payload


def load_image_dimensions(images_dir: Path) -> dict[str, tuple[int, int]]:
	"""Read image dimensions by exact filename for CVAT image records."""
	dimensions: dict[str, tuple[int, int]] = {}
	if not images_dir.exists():
		return dimensions

	try:
		from PIL import Image  # type: ignore
	except Exception:
		Image = None  # type: ignore[assignment]

	try:
		import cv2  # type: ignore
	except Exception:
		cv2 = None  # type: ignore[assignment]

	for image_path in images_dir.rglob("*"):
		if not image_path.is_file():
			continue
		try:
			if Image is not None:
				with Image.open(image_path) as image:
					dimensions[image_path.name] = image.size
				continue
			if cv2 is not None:
				frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
				if frame is not None:
					height, width = frame.shape[:2]
					dimensions[image_path.name] = (width, height)
		except Exception:
			continue
	return dimensions


def validate_unique_images(entries: Iterable[dict[str, Any]]) -> None:
	"""Reject payloads that reference the same image filename more than once."""
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
		raise ValueError(f"Duplicate image names found in predictions.json: {duplicate_list}")


def parse_prediction(raw: Any) -> NormalizedPrediction | None:
	"""Normalize a single raw prediction into CVAT box coordinates."""
	if not isinstance(raw, dict):
		warnings.warn(f"Skipping invalid prediction entry: expected object, got {type(raw).__name__}")
		return None

	bbox = raw.get("bbox")
	if not isinstance(bbox, dict):
		warnings.warn("Skipping invalid prediction entry: missing bbox object")
		return None

	required_keys = {"x", "y", "width", "height"}
	if not required_keys.issubset(bbox):
		warnings.warn("Skipping invalid prediction entry: bbox missing required fields")
		return None

	text = raw.get("text")
	if not isinstance(text, str):
		warnings.warn("Skipping invalid prediction entry: missing text field")
		return None

	try:
		x = float(bbox["x"])
		y = float(bbox["y"])
		width = float(bbox["width"])
		height = float(bbox["height"])
	except (TypeError, ValueError):
		warnings.warn("Skipping invalid prediction entry: bbox values must be numeric")
		return None

	if any(value < 0 for value in (x, y, width, height)):
		warnings.warn("Skipping invalid prediction entry: bbox values must be non-negative")
		return None
	if width <= 0 or height <= 0:
		warnings.warn("Skipping invalid prediction entry: bbox width and height must be greater than zero")
		return None

	return NormalizedPrediction(x=x, y=y, width=width, height=height, text=text)


def normalize_entries(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], ValidationStats]:
	"""Validate and normalize predictions while preserving every image row."""
	stats = ValidationStats(total_images=len(entries))
	normalized: list[dict[str, Any]] = []

	validate_unique_images(entries)

	for entry in entries:
		image_name = entry.get("image")
		if not isinstance(image_name, str) or not image_name.strip():
			warnings.warn("Skipping entry with missing or invalid image name")
			stats.invalid_predictions_skipped += 1
			continue

		predictions_raw = entry.get("predictions", [])
		if not isinstance(predictions_raw, list):
			warnings.warn(f"Skipping invalid predictions array for image {image_name}")
			predictions_raw = []

		normalized_predictions: list[NormalizedPrediction] = []
		for raw_prediction in predictions_raw:
			parsed = parse_prediction(raw_prediction)
			if parsed is None:
				stats.invalid_predictions_skipped += 1
				continue
			normalized_predictions.append(parsed)
			stats.total_predictions += 1

		if not normalized_predictions:
			stats.images_without_detections += 1

		normalized.append(
			{
				"image": image_name,
				"predictions": normalized_predictions,
			}
		)

	return normalized, stats


def build_cvat_xml(
	entries: list[dict[str, Any]],
	images_dir: Path,
	image_dimensions: dict[str, tuple[int, int]],
) -> ET.Element:
	"""Build a native CVAT image-annotation XML tree."""
	root = ET.Element("annotations")
	meta = ET.SubElement(root, "meta")
	task = ET.SubElement(meta, "task")
	ET.SubElement(task, "id").text = "0"
	ET.SubElement(task, "name").text = images_dir.name or "predictions"
	ET.SubElement(task, "size").text = str(len(entries))
	ET.SubElement(task, "mode").text = "annotation"
	ET.SubElement(task, "overlap").text = "0"
	ET.SubElement(task, "bugtracker").text = ""
	ET.SubElement(task, "flipped").text = "False"

	labels = ET.SubElement(task, "labels")
	label = ET.SubElement(labels, "label")
	ET.SubElement(label, "name").text = LICENSE_PLATE_LABEL
	label_attributes = ET.SubElement(label, "attributes")
	attribute = ET.SubElement(label_attributes, "attribute")
	ET.SubElement(attribute, "name").text = TEXT_ATTRIBUTE
	ET.SubElement(attribute, "mutable").text = "true"
	ET.SubElement(attribute, "input_type").text = "text"
	ET.SubElement(attribute, "default").text = ""
	ET.SubElement(attribute, "values").text = ""

	ET.SubElement(task, "segments")
	ET.SubElement(task, "owner")
	ET.SubElement(task, "source").text = "manual"

	ET.SubElement(meta, "dumped").text = datetime.now(timezone.utc).isoformat()

	for index, entry in enumerate(entries):
		image_name = entry["image"]
		predictions = entry["predictions"]
		width, height = image_dimensions.get(image_name, (0, 0))
		if width <= 0 or height <= 0:
			warnings.warn(f"Could not determine dimensions for image {image_name}; writing 0x0 in XML")

		image_element = ET.SubElement(
			root,
			"image",
			{
				"id": str(index),
				"name": image_name,
				"width": str(width),
				"height": str(height),
			},
		)

		for prediction in predictions:
			box = ET.SubElement(
				image_element,
				"box",
				{
					"label": LICENSE_PLATE_LABEL,
					"occluded": "0",
					"xtl": f"{prediction.x}",
					"ytl": f"{prediction.y}",
					"xbr": f"{prediction.xbr}",
					"ybr": f"{prediction.ybr}",
					"z_order": "0",
				},
			)
			attribute = ET.SubElement(box, "attribute", {"name": TEXT_ATTRIBUTE})
			attribute.text = prediction.text

	return root


def write_cvat_xml(root: ET.Element, output_path: Path) -> None:
	"""Persist the generated CVAT XML to disk."""
	tree = ET.ElementTree(root)
	ET.indent(tree, space="  ")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	tree.write(output_path, encoding="utf-8", xml_declaration=True)


def convert_predictions_to_cvat_xml(
	predictions_path: Path,
	output_path: Path,
	images_dir: Path,
) -> ValidationStats:
	"""Convert predictions JSON into a CVAT XML file and return validation stats."""
	entries = load_predictions(predictions_path)
	normalized_entries, stats = normalize_entries(entries)
	image_dimensions = load_image_dimensions(images_dir)
	root = build_cvat_xml(normalized_entries, images_dir, image_dimensions)
	write_cvat_xml(root, output_path)
	return stats


def main() -> int:
	"""CLI entry point for exporting CVAT pre-annotations."""
	parser = argparse.ArgumentParser(
		description="Convert predictions.json into a CVAT-compatible XML annotation file."
	)
	parser.add_argument(
		"--predictions",
		type=Path,
		default=DEFAULT_PREDICTIONS_PATH,
		help="Path to predictions.json.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT_PATH,
		help="Path to write the CVAT XML file.",
	)
	parser.add_argument(
		"--images-dir",
		type=Path,
		default=DEFAULT_IMAGES_DIR,
		help="Directory containing the source images used to infer image dimensions.",
	)
	args = parser.parse_args()

	predictions_path = args.predictions.resolve()
	output_path = args.output.resolve()
	images_dir = args.images_dir.resolve()

	try:
		stats = convert_predictions_to_cvat_xml(predictions_path, output_path, images_dir)
	except Exception as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 1

	print(f"Total images processed: {stats.total_images}")
	print(f"Total predictions converted: {stats.total_predictions}")
	print(f"Images without detections: {stats.images_without_detections}")
	print(f"Invalid predictions skipped: {stats.invalid_predictions_skipped}")
	print(f"Output file location: {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
