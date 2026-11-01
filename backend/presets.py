"""Opinionated starting points, Topaz-style: pick a preset for the kind
of source you have instead of choosing a model and filters by hand.

A preset is just a bundle of form values; the frontend fills the
controls from it and the user can still tweak anything. `engines` is a
preference list — the first one available on this machine is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Preset:
    name: str
    label: str
    description: str
    engines: tuple[str, ...]
    scale: int
    filters: dict = field(default_factory=dict)
    encode: dict = field(default_factory=dict)
    #: {"engine", "factor", "mode": "smooth" | "slowmo"} or empty for off
    interpolate: dict = field(default_factory=dict)

    def to_public_dict(self) -> dict:
        return asdict(self)


_ANIME_VIDEO = ("anime4k-v4", "realesrgan-animevideov3-onnx", "realesrgan-animevideov3")
_RIFE = "rife-v4.6"

PRESETS: list[Preset] = [
    Preset(
        name="anime-hd-to-4k",
        label="Anime — 1080p to 4K",
        description="Clean modern anime, web or Blu-ray. Video-trained model, no extra processing.",
        engines=_ANIME_VIDEO,
        scale=2,
    ),
    Preset(
        name="anime-web-cleanup",
        label="Anime — compressed web rip",
        description="Blocky, over-compressed source. Light denoise before the model so it doesn't sharpen the artifacts.",
        engines=_ANIME_VIDEO,
        scale=2,
        filters={"denoise": 1},
    ),
    Preset(
        name="anime-dvd",
        label="Anime — DVD / old TV (480p)",
        description="Interlaced standard definition. Deinterlace, medium denoise, then 4x to 1080p-class.",
        engines=_ANIME_VIDEO,
        scale=4,
        filters={"deinterlace": True, "denoise": 2},
    ),
    Preset(
        name="anime-max-detail",
        label="Anime — max line detail (stills model)",
        description="Crispest line art from the image-trained model; can flicker on noisy sources. 4x only.",
        engines=("realesrgan-x4plus-anime",),
        scale=4,
        filters={"denoise": 1},
    ),
    Preset(
        name="live-action-4x",
        label="Live action — 4x general model",
        description="Real footage rather than animation: the general RRDB model. Slow, but the right tool for photos, film and game captures.",
        engines=("realesrgan-x4plus",),
        scale=4,
        filters={"denoise": 1},
    ),
    Preset(
        name="anime-4k-60fps",
        label="Anime — 4K + smooth 2x frame rate",
        description="Upscale and RIFE-double the frame rate (24 → 48 fps) for smoother motion. Audio is kept.",
        engines=_ANIME_VIDEO,
        scale=2,
        interpolate={"engine": _RIFE, "factor": 2, "mode": "smooth"},
    ),
    Preset(
        name="slow-motion-4x",
        label="Slow motion — 4x, no upscale",
        description="RIFE generates three frames between every pair; playback at the original frame rate makes it four times slower. Audio is dropped.",
        engines=("none",),
        scale=1,
        interpolate={"engine": _RIFE, "factor": 4, "mode": "slowmo"},
    ),
    Preset(
        name="archive-hevc",
        label="Archive — HEVC, smaller files",
        description="Same anime video model, encoded to HEVC at high quality for long-term storage.",
        engines=_ANIME_VIDEO,
        scale=2,
        encode={"codec": "hevc", "quality": 20},
    ),
]
