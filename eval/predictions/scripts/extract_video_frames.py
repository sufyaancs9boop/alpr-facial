from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
SUPPORTED_FORMATS = {"jpg", "png"}
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "dataset" / "videos-alpr"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "dataset" / "videos-frames"


@dataclass
class VideoResult:
    """Summary for a single processed video."""

    video_name: str
    duration_seconds: float | None
    frames_extracted: int
    elapsed_seconds: float
    skipped: bool = False
    failed: bool = False


def ensure_ffmpeg_available() -> None:
    """Verify that ffmpeg is installed and available on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg was not found on PATH. Install FFmpeg before running this script.")


def ensure_input_directory(input_dir: Path) -> None:
    """Validate the input directory exists."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")


def ensure_output_directory(output_dir: Path) -> None:
    """Create the output directory if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)


def iter_supported_videos(input_dir: Path) -> Iterable[Path]:
    """Yield supported video files from the top level of the input directory."""
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def probe_duration_seconds(video_path: Path) -> float | None:
    """Return the video duration in seconds if ffprobe is available."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None

    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def build_output_pattern(output_dir: Path, video_path: Path, output_format: str) -> str:
    """Build the ffmpeg output filename pattern for a specific video."""
    return str(output_dir / f"{video_path.stem}_frame%06d.{output_format}")


def existing_frame_files(output_dir: Path, video_path: Path, output_format: str) -> list[Path]:
    """Return already-existing extracted frames for the given video."""
    prefix = f"{video_path.stem}_frame"
    suffix = f".{output_format.lower()}"
    return sorted(
        path for path in output_dir.iterdir()
        if path.is_file() and path.name.startswith(prefix) and path.name.endswith(suffix)
    )


def remove_existing_frames(output_dir: Path, video_path: Path, output_format: str) -> None:
    """Delete already-existing extracted frames for the given video."""
    for path in existing_frame_files(output_dir, video_path, output_format):
        path.unlink()


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float,
    output_format: str,
    overwrite: bool,
) -> VideoResult:
    """Extract frames from a single video using ffmpeg."""
    duration_seconds = probe_duration_seconds(video_path)
    start_time = time.perf_counter()

    existing_frames = existing_frame_files(output_dir, video_path, output_format)
    if existing_frames and not overwrite:
        elapsed_seconds = time.perf_counter() - start_time
        return VideoResult(
            video_name=video_path.name,
            duration_seconds=duration_seconds,
            frames_extracted=0,
            elapsed_seconds=elapsed_seconds,
            skipped=True,
        )

    if overwrite:
        remove_existing_frames(output_dir, video_path, output_format)

    frame_pattern = build_output_pattern(output_dir, video_path, output_format)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-start_number",
        "1",
        "-vf",
        f"fps={fps}",
    ]
    if output_format == "jpg":
        command.extend(["-q:v", "2"])
    if overwrite:
        command.append("-y")
    else:
        command.append("-n")
    command.append(frame_pattern)

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed_seconds = time.perf_counter() - start_time
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffmpeg failed for {video_path.name}")

    frames_extracted = len(existing_frame_files(output_dir, video_path, output_format))
    if overwrite:
        # When overwriting, frames are newly created; counting by prefix is reliable.
        frames_extracted = len(existing_frame_files(output_dir, video_path, output_format))

    return VideoResult(
        video_name=video_path.name,
        duration_seconds=duration_seconds,
        frames_extracted=frames_extracted,
        elapsed_seconds=elapsed_seconds,
    )


def _format_duration(duration_seconds: float | None) -> str:
    """Format an optional duration for logging."""
    if duration_seconds is None:
        return "unknown"
    return f"{duration_seconds:.2f}s"


def run(input_dir: Path, output_dir: Path, fps: float, output_format: str, overwrite: bool) -> int:
    """Run frame extraction across all supported videos in the input directory."""
    ensure_ffmpeg_available()
    ensure_input_directory(input_dir)
    ensure_output_directory(output_dir)

    videos = list(iter_supported_videos(input_dir))
    processed = 0
    skipped = 0
    failed = 0
    total_frames = 0

    for video_path in videos:
        try:
            result = extract_frames(video_path, output_dir, fps, output_format, overwrite)
            if result.skipped:
                skipped += 1
                print(
                    f"{result.video_name} | duration={_format_duration(result.duration_seconds)} | "
                    f"frames=0 | time={result.elapsed_seconds:.2f}s | skipped(existing frames)"
                )
                continue

            processed += 1
            total_frames += result.frames_extracted
            print(
                f"{result.video_name} | duration={_format_duration(result.duration_seconds)} | "
                f"frames={result.frames_extracted} | time={result.elapsed_seconds:.2f}s"
            )
        except Exception as exc:
            failed += 1
            print(f"{video_path.name} | failed: {exc}", file=sys.stderr)

    print(f"Videos processed: {processed}")
    print(f"Videos skipped: {skipped}")
    print(f"Total frames extracted: {total_frames}")
    print(f"Failed videos: {failed}")
    print(f"Output directory: {output_dir}")
    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the frame extractor."""
    parser = argparse.ArgumentParser(description="Extract frames from ALPR evaluation videos using FFmpeg.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR, help="Input directory containing videos.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for extracted frames.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract from each video.")
    parser.add_argument(
        "--format",
        choices=sorted(SUPPORTED_FORMATS),
        default="jpg",
        help="Output image format for extracted frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted frames for a video.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point for frame extraction."""
    args = parse_args()
    return run(args.input.resolve(), args.output.resolve(), args.fps, args.format.lower(), args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())