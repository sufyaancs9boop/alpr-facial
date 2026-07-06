from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from services.alpr_service import AlprService

FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_INPUT_DIR = ROOT / "eval" / "dataset" / "videos-frames"
DEFAULT_OUTPUT_PATH = ROOT / "eval" / "predictions" / "predictions_frames.json"


async def _noop(*args, **kwargs):
	return None


def _build_service() -> AlprService:
	"""Create a prediction-only ALPR service with side effects disabled."""
	service = AlprService(None, None, None, None, None, None)
	service._process_and_log = _noop  # type: ignore[attr-defined]
	service._log_and_alert = _noop  # type: ignore[attr-defined]
	service._save_face_event = _noop  # type: ignore[attr-defined]
	return service


def _iter_frames(frames_dir: Path) -> Iterable[Path]:
	"""Yield every extracted frame file in lexical order."""
	for path in sorted(frames_dir.iterdir()):
		if path.is_file() and path.suffix.lower() in FRAME_EXTENSIONS:
			yield path


def _plate_to_prediction(plate) -> dict[str, Any]:
	"""Convert a plate result into the exported JSON structure."""
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


async def export_frame_predictions(frames_dir: Path, output_path: Path) -> list[dict[str, Any]]:
	"""Run backend inference on each extracted frame and export raw predictions."""
	service = _build_service()
	results: list[dict[str, Any]] = []

	try:
		for frame_path in _iter_frames(frames_dir):
			try:
				detection = await service.detect_from_bytes(
					frame_path.read_bytes(),
					generate_thumbnail=False,
				)
				predictions = [_plate_to_prediction(plate) for plate in detection.plates]
			except asyncio.CancelledError:
				raise
			except Exception as exc:
				print(f"Warning: failed to process {frame_path.name}: {exc}", file=sys.stderr)
				predictions = []

			results.append(
				{
					"image": frame_path.relative_to(frames_dir).as_posix(),
					"predictions": predictions,
				}
			)
	finally:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
	return results


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for the frame exporter."""
	parser = argparse.ArgumentParser(description="Export backend predictions for extracted video frames.")
	parser.add_argument(
		"--input",
		type=Path,
		default=DEFAULT_INPUT_DIR,
		help="Directory containing extracted video frames.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT_PATH,
		help="Path to write predictions_frames.json.",
	)
	return parser.parse_args()


def main() -> int:
	"""CLI entry point for exporting frame predictions."""
	args = parse_args()
	frames_dir = args.input.resolve()
	output_path = args.output.resolve()

	if not frames_dir.exists() or not frames_dir.is_dir():
		print(f"Error: input directory does not exist: {frames_dir}", file=sys.stderr)
		return 1

	try:
		asyncio.run(export_frame_predictions(frames_dir, output_path))
	except KeyboardInterrupt:
		print(f"Interrupted. Partial predictions were saved to {output_path}")
		return 130

	print(f"Saved predictions to {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())