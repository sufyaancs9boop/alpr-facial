from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from services.alpr_service import AlprService

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".dng"}


async def _noop(*args, **kwargs):
	return None


def _build_service() -> AlprService:
	service = AlprService(None, None, None, None, None, None)
	service._process_and_log = _noop  # type: ignore[attr-defined]
	service._log_and_alert = _noop  # type: ignore[attr-defined]
	service._save_face_event = _noop  # type: ignore[attr-defined]
	return service


def _iter_images(dataset: Path):
	for path in sorted(dataset.rglob("*")):
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
			yield path


def _plate_to_prediction(plate) -> dict[str, Any]:
	bbox = plate.bounding_box
	return {
		"bbox": {
			"x": bbox["x"],
			"y": bbox["y"],
			"width": bbox["width"],
			"height": bbox["height"],
		},
		"text": plate.text,
		"confidence": plate.confidence,
	}


async def export_predictions(dataset: Path, output: Path) -> list[dict[str, Any]]:
	service = _build_service()
	results: list[dict[str, Any]] = []

	for image_path in _iter_images(dataset):
		try:
			detection = await service.detect_from_bytes(
				image_path.read_bytes(),
				generate_thumbnail=False,
			)
			predictions = [_plate_to_prediction(plate) for plate in detection.plates]
		except Exception:
			predictions = []

		results.append({
			"image": image_path.relative_to(dataset).as_posix(),
			"predictions": predictions,
		})

	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text(json.dumps(results, indent=2), encoding="utf-8")
	return results


def main() -> int:
	parser = argparse.ArgumentParser(description="Export raw ALPR predictions for image datasets.")
	parser.add_argument(
		"--dataset",
		type=Path,
		default=ROOT / "eval" / "dataset" / "images-alpr",
		help="Directory containing the image dataset.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=ROOT / "eval" / "predictions" / "predictions.json",
		help="Path to write the predictions JSON file.",
	)
	args = parser.parse_args()

	dataset = args.dataset.resolve()
	output = args.output.resolve()
	asyncio.run(export_predictions(dataset, output))
	print(f"Saved predictions to {output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
