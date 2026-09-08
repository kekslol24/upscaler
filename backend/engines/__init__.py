"""Engine registry — the harness's plug-in point.

To add a new model backend: implement UpscaleEngine in a new module in
this package, then add an instance to REGISTRY below.
"""

from __future__ import annotations

from .base import UpscaleEngine
from .realesrgan_ncnn import RealEsrganNcnnEngine

REGISTRY: dict[str, UpscaleEngine] = {
    engine.name: engine
    for engine in [
        RealEsrganNcnnEngine(
            name="realesrgan-animevideov3",
            label="Real-ESRGAN — anime video",
            model_name="realesr-animevideov3",
            description="Trained on video frames; best default for anime episodes/clips.",
            supported_scales=(2, 3, 4),
        ),
        RealEsrganNcnnEngine(
            name="realesrgan-x4plus-anime",
            label="Real-ESRGAN — anime, max detail",
            model_name="realesrgan-x4plus-anime",
            description="Trained on anime stills; crisper line art, may flicker slightly frame-to-frame.",
            supported_scales=(4,),
        ),
    ]
}


def get_engine(name: str) -> UpscaleEngine:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown engine: {name!r}") from None
