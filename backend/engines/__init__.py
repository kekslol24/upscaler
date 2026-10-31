"""Engine registry — the harness's plug-in point.

To add a new model backend: implement UpscaleEngine (or FrameInterpolator)
in a new module in this package, then add an instance to the matching
registry below.
"""

from __future__ import annotations

from .base import FrameInterpolator, UpscaleEngine
from .realesrgan_ncnn import RealEsrganNcnnEngine
from .realesrgan_onnx import RealEsrganOnnxEngine
from .rife_ncnn import RifeNcnnInterpolator
from .rife_torch import RifeTorchInterpolator

#: pseudo-engine name for "interpolate only, don't upscale"
NO_UPSCALE = "none"

# Order matters: the frontend defaults to the first available engine.
REGISTRY: dict[str, UpscaleEngine] = {
    engine.name: engine
    for engine in [
        RealEsrganOnnxEngine(
            name="realesrgan-animevideov3-onnx",
            label="Real-ESRGAN — anime video (fast, DirectML/CUDA)",
            model_prefix="realesr-animevideov3",
            description="Same model and output as the Vulkan engine, several times faster: whole-frame GPU inference, streamed straight between decoder and encoder.",
            supported_scales=(2, 3, 4),
        ),
        RealEsrganNcnnEngine(
            name="realesrgan-animevideov3",
            label="Real-ESRGAN — anime video (Vulkan)",
            model_name="realesr-animevideov3",
            description="Trained on video frames; best default for anime episodes/clips. Portable Vulkan build, slower than the DirectML engine.",
            supported_scales=(2, 3, 4),
        ),
        RealEsrganNcnnEngine(
            name="realesrgan-x4plus-anime",
            label="Real-ESRGAN — anime, max detail",
            model_name="realesrgan-x4plus-anime",
            description="Trained on anime stills; crisper line art, may flicker slightly frame-to-frame.",
            supported_scales=(4,),
        ),
        RealEsrganNcnnEngine(
            name="realesrgan-x4plus",
            label="Real-ESRGAN — live action / general",
            model_name="realesrgan-x4plus",
            description="The general-purpose RRDB model for real footage: photos, live action, games. Much slower than the anime models. 4x only.",
            supported_scales=(4,),
        ),
    ]
}

INTERPOLATORS: dict[str, FrameInterpolator] = {
    interp.name: interp
    for interp in [
        RifeTorchInterpolator(
            name="rife-v4.6",
            label="RIFE v4.6 (CUDA/ROCm)",
            description="Optical-flow frame interpolation on the GPU via PyTorch (CUDA on NVIDIA, ROCm on AMD), streamed between decoder and encoder. Much faster than the Vulkan build at 4K.",
        ),
        RifeNcnnInterpolator(
            name="rife-v4.6-vulkan",
            label="RIFE v4.6 (Vulkan)",
            model_name="rife-v4.6",
            description="Portable Vulkan build of RIFE v4.6. Falls back to directory-mode PNG processing.",
        ),
    ]
}


def get_engine(name: str) -> UpscaleEngine:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown engine: {name!r}") from None


def get_interpolator(name: str) -> FrameInterpolator:
    try:
        return INTERPOLATORS[name]
    except KeyError:
        raise ValueError(f"unknown interpolator: {name!r}") from None
