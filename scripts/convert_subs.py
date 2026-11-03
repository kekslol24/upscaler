"""Convert subtitle tracks in existing videos without re-encoding.

The upscaler's "Subtitles" option applies when rendering. This script does
the same conversion for files that are already encoded: video and audio are
stream-copied, only the subtitle tracks change.

    .venv/bin/python scripts/convert_subs.py --in-place "/media/anime/Show/Season 1"
    .venv/bin/python scripts/convert_subs.py --format ass --output-dir out video.mkv

SRT is the format Android and Android TV can render client-side, so a
4K file with an embedded ASS track can direct-play instead of forcing the
server to burn the subtitles in. Image subtitles (PGS/DVD) cannot become
SRT and are dropped by --format srt; use --format ass to keep them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import pipeline  # noqa: E402

VIDEO_EXTS = {
    ".mkv", ".mp4", ".m4v", ".mov", ".avi", ".webm", ".ts", ".m2ts",
    ".mpg", ".mpeg", ".wmv",
}
#: text codecs that already count as the target format
_TARGET_CODECS = {
    "srt": {"subrip", "srt"},
    "ass": {"ass", "ssa"},
}


def _iter_videos(paths: list[str]):
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                relative = candidate.relative_to(path)
                if any(part.startswith(".") for part in relative.parts):
                    continue
                if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTS:
                    yield candidate
        elif path.is_file():
            yield path
        else:
            print(f"skip (not found): {path}")


def _av_codecs(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    out = pipeline.proc.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type,codec_name",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    streams = json.loads(out).get("streams", [])
    video = tuple(s.get("codec_name", "") for s in streams if s.get("codec_type") == "video")
    audio = tuple(s.get("codec_name", "") for s in streams if s.get("codec_type") == "audio")
    return video, audio


def _verify(source: Path, output: Path) -> str | None:
    """Return an error string when the remux lost or changed the A/V."""
    src = pipeline.probe(source)
    out = pipeline.probe(output)
    if (src.width, src.height) != (out.width, out.height):
        return f"resolution changed ({src.width}x{src.height} -> {out.width}x{out.height})"
    if src.has_audio != out.has_audio:
        return "audio stream lost"
    tolerance = max(1.0, src.duration * 0.002)
    if abs(src.duration - out.duration) > tolerance:
        return f"duration changed ({src.duration:.2f}s -> {out.duration:.2f}s)"
    if _av_codecs(source) != _av_codecs(output):
        return "video/audio streams changed"
    if not pipeline.has_subtitles(output):
        return "output has no subtitle stream"
    return None


def _target_path(source: Path, args: argparse.Namespace) -> Path:
    if args.in_place:
        if source.suffix.lower() != ".mkv":
            raise ValueError("--in-place only supports MKV sources; use --output-dir")
        return source
    out_dir = Path(args.output_dir) if args.output_dir else source.parent
    suffix = args.suffix or f"_{args.format}"
    return out_dir / f"{source.stem}{suffix}.mkv"


def _convert(source: Path, args: argparse.Namespace) -> bool:
    subs = pipeline.probe_subtitles(source)
    text_subs = [s for s in subs if s["codec"] in pipeline.TEXT_SUBTITLE_CODECS]
    if not text_subs:
        reason = "no text subtitles" if not subs else "only image subtitles"
        print(f"skip ({reason}): {source}")
        return False
    if not args.force and all(
        s["codec"] in _TARGET_CODECS[args.format] for s in text_subs
    ):
        print(f"skip (already {args.format.upper()}): {source}")
        return False

    try:
        target = _target_path(source, args)
    except ValueError as exc:
        print(f"skip ({exc}): {source}")
        return False

    if not args.in_place:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not args.overwrite:
            print(f"skip (exists, use --overwrite): {target}")
            return False
        if target.resolve() == source.resolve():
            print(f"skip (output would overwrite the source): {source}")
            return False

    print(f"{source} -> {target} ({args.format})")
    if args.dry_run:
        return True

    # Write beside the source so the final rename stays on one filesystem.
    temp = target.with_name(f".{target.stem}.subs-tmp{target.suffix}")
    temp.unlink(missing_ok=True)
    try:
        pipeline.mux_subtitles(
            source, source, temp, subtitle_format=args.format,
        )
        error = _verify(source, temp)
        if error:
            raise RuntimeError(error)
        os.replace(temp, target)
    except Exception as exc:  # noqa: BLE001 - report and keep the original
        temp.unlink(missing_ok=True)
        print(f"failed: {source}: {exc}")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert text subtitle tracks in existing videos without re-encoding.",
    )
    parser.add_argument("paths", nargs="+", help="video files or directories")
    parser.add_argument(
        "--format", choices=("srt", "ass"), default="srt",
        help="target subtitle format (default: srt)",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="replace each MKV after verifying the result (no backup)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="write converted files here instead of beside the source",
    )
    parser.add_argument(
        "--suffix", help="filename suffix for non-in-place output (default: _srt/_ass)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="overwrite existing output files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="reconvert even if the subtitle is already in the target format",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="show what would be converted without writing anything",
    )
    args = parser.parse_args(argv)

    if args.in_place and args.output_dir:
        parser.error("--in-place and --output-dir are mutually exclusive")

    videos = list(_iter_videos(args.paths))
    if not videos:
        print("no video files found")
        return 1

    converted = sum(1 for video in videos if _convert(video, args))
    verb = "would convert" if args.dry_run else "converted"
    print(f"{verb} {converted} of {len(videos)} file(s)")
    return 0 if converted or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
