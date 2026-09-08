"""ffprobe/ffmpeg helpers: extract frames from a video, and reassemble
upscaled frames back into a video with the original audio track and
frame rate.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FfmpegNotFound(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _require_ffmpeg() -> None:
    if not ffmpeg_available():
        raise FfmpegNotFound("ffmpeg/ffprobe not found on PATH")


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    has_audio: bool
    duration: float


def _parse_frame_rate(rate: str) -> float:
    if "/" in rate:
        num, den = rate.split("/")
        den = float(den)
        return float(num) / den if den else 0.0
    return float(rate)


def probe(input_path: Path) -> VideoInfo:
    _require_ffmpeg()
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(input_path),
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    video_stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    return VideoInfo(
        fps=_parse_frame_rate(video_stream["r_frame_rate"]),
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        has_audio=has_audio,
        duration=float(data["format"].get("duration", 0.0)),
    )


def extract_frames(input_path: Path, frames_dir: Path) -> int:
    """Extract every frame of input_path as PNGs into frames_dir.
    Returns the number of frames extracted.
    """
    _require_ffmpeg()
    frames_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(input_path),
            "-start_number", "0",
            str(frames_dir / "%08d.png"),
        ],
        check=True,
    )
    return sum(1 for _ in frames_dir.iterdir())


def reassemble(
    frames_dir: Path,
    output_path: Path,
    fps: float,
    audio_source: Path | None,
) -> None:
    """Reassemble PNG frames in frames_dir into output_path at fps,
    muxing in the audio track from audio_source if given.
    """
    _require_ffmpeg()
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", str(frames_dir / "%08d.png"),
    ]
    if audio_source is not None:
        cmd += ["-i", str(audio_source)]

    cmd += ["-map", "0:v"]
    if audio_source is not None:
        cmd += ["-map", "1:a", "-c:a", "copy", "-shortest"]

    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path)]
    subprocess.run(cmd, check=True)
