"""ffprobe/ffmpeg helpers.

Two ways to move frames through an engine:

- directory mode: extract_frames() writes every frame as a PNG, the
  engine processes the directory, reassemble() encodes the result.
- streaming mode: open_frame_reader() / open_frame_writer() hand the
  engine raw RGB frames over pipes so decode, upscale and encode overlap
  and nothing touches the disk. Same encoder settings either way.

RenderOptions carries everything about a render that is not the model:
which part of the source to process (Clip), ffmpeg filters applied
before/after the model (Filters), and how to encode (EncodeSettings).
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from .paths import DATA_DIR


class FfmpegNotFound(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _require_ffmpeg() -> None:
    if not ffmpeg_available():
        raise FfmpegNotFound("ffmpeg/ffprobe not found on PATH")


# -- render options ---------------------------------------------------------

@dataclass(frozen=True)
class Clip:
    """A time window of the source, for previews."""
    start: float
    duration: float


@dataclass(frozen=True)
class Filters:
    """ffmpeg filters around the model. Pre-filters run on the decoded
    source (so the model sees a cleaner, progressive frame); post-filters
    run on the upscaled output."""
    deinterlace: bool = False
    #: 0 = off, 1 = light, 2 = medium, 3 = strong (hqdn3d)
    denoise: int = 0
    sharpen: bool = False

    def pre(self) -> list[str]:
        vf = []
        if self.deinterlace:
            vf.append("bwdif=mode=send_frame")
        if self.denoise:
            vf.append(_DENOISE_LEVELS[min(self.denoise, 3)])
        return vf

    def post(self) -> list[str]:
        return ["unsharp=5:5:0.6:5:5:0.0"] if self.sharpen else []


_DENOISE_LEVELS = {
    1: "hqdn3d=2:1.5:3:2.25",
    2: "hqdn3d=4:3:6:4.5",
    3: "hqdn3d=7:5:10:7.5",
}


@dataclass(frozen=True)
class EncodeSettings:
    codec: str = "h264"          # "h264" | "hevc"
    encoder: str = "auto"        # "auto" or an ffmpeg encoder name
    #: CRF / CQ style quality, lower is better; 18 is visually lossless-ish
    quality: int = 18


@dataclass(frozen=True)
class RenderOptions:
    clip: Clip | None = None
    filters: Filters = field(default_factory=Filters)
    encode: EncodeSettings = field(default_factory=EncodeSettings)


# -- encoders ---------------------------------------------------------------

#: per codec, best-first. GPU encoders are only used if a test encode works.
_ENCODER_CANDIDATES = {
    "h264": ["h264_nvenc", "h264_amf", "h264_qsv", "h264_videotoolbox", "libx264"],
    "hevc": ["hevc_nvenc", "hevc_amf", "hevc_qsv", "hevc_videotoolbox", "libx265"],
}

ENCODER_LABELS = {
    "h264_nvenc": "H.264 — NVIDIA NVENC",
    "hevc_nvenc": "HEVC — NVIDIA NVENC",
    "h264_amf": "H.264 — AMD AMF",
    "hevc_amf": "HEVC — AMD AMF",
    "h264_qsv": "H.264 — Intel Quick Sync",
    "hevc_qsv": "HEVC — Intel Quick Sync",
    "h264_videotoolbox": "H.264 — Apple VideoToolbox",
    "hevc_videotoolbox": "HEVC — Apple VideoToolbox",
    "libx264": "H.264 — libx264 (CPU)",
    "libx265": "HEVC — libx265 (CPU)",
}


def _encoder_works(name: str) -> bool:
    """A listed encoder is not a usable one (the driver may be missing), so
    actually encode a few tiny frames with it."""
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostdin",
                "-f", "lavfi", "-i", "color=black:s=64x64:r=10:d=0.3",
                "-c:v", name, "-pix_fmt", "yuv420p", "-f", "null", "-",
            ],
            capture_output=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def _ffmpeg_version() -> str:
    return subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]


@functools.lru_cache(maxsize=1)
def available_encoders() -> list[str]:
    """Encoders that actually work on this machine, per-codec best first.

    The test encodes take a few seconds for GPU encoders, so the result is
    cached on disk keyed by the ffmpeg build (a driver change that breaks
    an encoder surfaces as a job error, and deleting the cache file resets).
    """
    if not ffmpeg_available():
        return []
    version = _ffmpeg_version()
    cache_path = DATA_DIR / "encoders.json"
    try:
        cached = json.loads(cache_path.read_text())
        if cached.get("ffmpeg") == version:
            return list(cached["encoders"])
    except (OSError, ValueError, KeyError):
        pass

    listed = subprocess.run(
        ["ffmpeg", "-v", "error", "-hide_banner", "-encoders"],
        capture_output=True, text=True,
    ).stdout
    working = []
    for candidates in _ENCODER_CANDIDATES.values():
        for name in candidates:
            if f" {name} " in listed and _encoder_works(name):
                working.append(name)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"ffmpeg": version, "encoders": working}))
    except OSError:
        pass
    return working


def encoder_codec(name: str) -> str:
    return "hevc" if name.startswith("hevc") or name == "libx265" else "h264"


def resolve_encoder(settings: EncodeSettings) -> str:
    if settings.encoder != "auto":
        return settings.encoder
    for name in _ENCODER_CANDIDATES.get(settings.codec, ()):
        if name in available_encoders():
            return name
    raise RuntimeError(f"no working {settings.codec} encoder found")


def encode_args(settings: EncodeSettings, container: str = "mp4") -> list[str]:
    """Video encoder arguments for the chosen encoder at the requested
    quality. Each encoder family spells constant-quality differently."""
    enc = resolve_encoder(settings)
    q = str(max(0, min(51, settings.quality)))
    args = ["-c:v", enc, "-pix_fmt", "yuv420p"]
    if enc in ("libx264", "libx265"):
        args += ["-preset", "medium", "-crf", q]
    elif enc.endswith("_nvenc"):
        args += ["-preset", "p5", "-rc", "vbr", "-cq", q, "-b:v", "0"]
    elif enc.endswith("_amf"):
        args += ["-quality", "quality", "-rc", "cqp", "-qp_i", q, "-qp_p", q]
    elif enc.endswith("_qsv"):
        args += ["-global_quality", q]
    elif enc.endswith("_videotoolbox"):
        args += ["-q:v", str(max(1, 100 - int(q) * 2))]
    if encoder_codec(enc) == "hevc" and container == "mp4":
        # Apple/Windows players only recognise HEVC in MP4 under this tag
        args += ["-tag:v", "hvc1"]
    return args


# -- audio --------------------------------------------------------------------

#: audio codecs that both fit in MP4 and play everywhere; anything else
#: (Opus/FLAC/DTS-heavy MKV sources) gets transcoded to AAC — a straight
#: copy either fails the mux or produces a file most players treat as silent
_MP4_SAFE_AUDIO = {"aac", "mp3", "ac3", "eac3"}


def _audio_args(audio_source: Path) -> list[str]:
    try:
        codec = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_source),
            ],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        codec = ""
    if codec in _MP4_SAFE_AUDIO:
        return ["-c:a", "copy"]
    return ["-c:a", "aac", "-b:a", "192k"]


def _clip_input_args(clip: Clip | None) -> list[str]:
    """`-ss`/`-t` as *input* options: ffmpeg seeks to the nearest keyframe
    and decodes forward, so the cut is frame-accurate and still fast."""
    if clip is None:
        return []
    return ["-ss", f"{clip.start:.3f}", "-t", f"{clip.duration:.3f}"]


def _vf_args(filters: list[str]) -> list[str]:
    return ["-vf", ",".join(filters)] if filters else []


# -- probing ------------------------------------------------------------------

@dataclass
class VideoInfo:
    fps: float
    #: exact frame rate as ffprobe reports it, e.g. "24000/1001"
    frame_rate: str
    width: int
    height: int
    has_audio: bool
    duration: float

    @property
    def estimated_frames(self) -> int:
        return round(self.duration * self.fps) if self.duration and self.fps else 0

    def frames_in(self, clip: Clip | None) -> int:
        if clip is None:
            return self.estimated_frames
        return round(min(clip.duration, max(self.duration - clip.start, 0)) * self.fps)


def scale_frame_rate(frame_rate: str, factor: int) -> str:
    """'24000/1001' x 2 -> '48000/1001', keeping the exact rational."""
    scaled = Fraction(frame_rate) * factor
    return f"{scaled.numerator}/{scaled.denominator}"


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
        frame_rate=video_stream["r_frame_rate"],
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        has_audio=has_audio,
        duration=float(data["format"].get("duration", 0.0)),
    )


# -- directory mode -----------------------------------------------------------

def extract_frames(
    input_path: Path,
    frames_dir: Path,
    options: RenderOptions | None = None,
) -> int:
    """Extract every frame of input_path (or of options.clip) as PNGs into
    frames_dir, with pre-filters applied. Returns the number of frames.
    """
    _require_ffmpeg()
    options = options or RenderOptions()
    frames_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            *_clip_input_args(options.clip),
            "-i", str(input_path),
            *_vf_args(options.filters.pre()),
            "-start_number", "0",
            # lossless either way; level 1 halves extraction time vs the
            # default at the cost of ~30% larger PNGs
            "-compression_level", "1",
            str(frames_dir / "%08d.png"),
        ],
        check=True,
    )
    return sum(1 for _ in frames_dir.iterdir())


def reassemble(
    frames_dir: Path,
    output_path: Path,
    fps: float | str,
    audio_source: Path | None,
    options: RenderOptions | None = None,
) -> None:
    """Reassemble PNG frames in frames_dir into output_path at fps,
    muxing in the audio track from audio_source if given.
    """
    _require_ffmpeg()
    options = options or RenderOptions()
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", str(frames_dir / "%08d.png"),
    ]
    if audio_source is not None:
        cmd += [*_clip_input_args(options.clip), "-i", str(audio_source)]

    cmd += ["-map", "0:v", *_vf_args(options.filters.post())]
    if audio_source is not None:
        cmd += ["-map", "1:a"] + _audio_args(audio_source)

    cmd += encode_args(options.encode, output_path.suffix.lstrip(".")) + [str(output_path)]
    subprocess.run(cmd, check=True)


def concat_segments(
    segments: list[Path],
    output_path: Path,
    audio_source: Path | None,
    options: RenderOptions | None = None,
) -> None:
    """Join video-only segment files (same codec/params) into output_path
    without re-encoding, muxing in audio from audio_source if given."""
    _require_ffmpeg()
    options = options or RenderOptions()
    list_file = output_path.with_suffix(".segments.txt")
    list_file.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in segments))
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(list_file)]
    if audio_source is not None:
        cmd += [*_clip_input_args(options.clip), "-i", str(audio_source)]
    cmd += ["-map", "0:v", "-c:v", "copy"]
    if audio_source is not None:
        cmd += ["-map", "1:a"] + _audio_args(audio_source)
    if encoder_codec(resolve_encoder(options.encode)) == "hevc" and output_path.suffix == ".mp4":
        cmd += ["-tag:v", "hvc1"]
    cmd.append(str(output_path))
    try:
        subprocess.run(cmd, check=True)
    finally:
        list_file.unlink(missing_ok=True)


# -- streaming mode -----------------------------------------------------------

def open_frame_reader(
    source: Path,
    options: RenderOptions | None = None,
) -> subprocess.Popen:
    """Start ffmpeg decoding `source` (a video, or an image-sequence
    pattern like frames/%08d.png) to raw RGB24 frames on its stdout,
    with options.clip and pre-filters applied."""
    _require_ffmpeg()
    options = options or RenderOptions()
    return subprocess.Popen(
        [
            "ffmpeg", "-v", "error",
            *_clip_input_args(options.clip),
            "-i", str(source),
            *_vf_args(options.filters.pre()),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def open_frame_writer(
    target: Path,
    width: int,
    height: int,
    frame_rate: str,
    audio_source: Path | None = None,
    options: RenderOptions | None = None,
) -> subprocess.Popen:
    """Start ffmpeg reading raw RGB24 frames of width x height on its
    stdin and encoding them to `target` (a video with options.encode, or
    PNGs if target is an image-sequence pattern)."""
    _require_ffmpeg()
    options = options or RenderOptions()
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-framerate", frame_rate,
        "-i", "-",
    ]
    if audio_source is not None:
        cmd += [*_clip_input_args(options.clip), "-i", str(audio_source)]
    cmd += ["-map", "0:v"]
    if audio_source is not None:
        cmd += ["-map", "1:a"] + _audio_args(audio_source)
    if str(target).endswith(".png"):
        cmd += ["-start_number", "0", "-compression_level", "1"]
    else:
        cmd += _vf_args(options.filters.post())
        cmd += encode_args(options.encode, target.suffix.lstrip("."))
    cmd.append(str(target))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def is_browser_playable(input_path: Path) -> bool:
    """8-bit H.264 in MP4/MOV is the one combination every browser plays."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,pix_fmt:format=format_name",
                "-of", "json", str(input_path),
            ],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return False
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format", {}).get("format_name", "")
    return (
        stream.get("codec_name") == "h264"
        and stream.get("pix_fmt") == "yuv420p"
        and ("mp4" in fmt or "mov" in fmt)
    )


def make_proxy(input_path: Path, output_path: Path, clip: Clip | None = None) -> None:
    """Quick browser-playable H.264 copy of the source (or of a clip of
    it) for the before/after view — the upload itself may be 10-bit HEVC
    in an MKV, which no browser plays."""
    _require_ffmpeg()
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            *_clip_input_args(clip),
            "-i", str(input_path),
            "-map", "0:v:0", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            str(output_path),
        ],
        check=True,
    )
