"""Real-ESRGAN engine, running on the prebuilt realesrgan-ncnn-vulkan binary.

This binary uses Vulkan compute directly (no PyTorch/CUDA), so it runs on
Intel/AMD/NVIDIA GPUs alike. scripts/setup.sh (Linux) or scripts/setup.ps1
(Windows) downloads it, along with all of its bundled models, into
engines_bin/realesrgan/.

The binary bundles several models; we expose the anime-relevant ones as
separate engine instances (see __init__.py) since they trade off
differently and both are worth having selectable:

- realesr-animevideov3: trained on video frames (not just stills), so it
  handles compression artifacts and frame-to-frame consistency better.
  Supports 2x/3x/4x. Best default for actual anime episodes/clips.
- realesrgan-x4plus-anime: trained on anime *images*, tends to produce
  crisper line art and flatter, cleaner colors, at the cost of being a
  still-image model (processed per-frame, so slightly more prone to
  frame-to-frame flicker on noisy sources). 4x only.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .. import proc as proc_
from ..paths import ENGINES_DIR
from .base import ProgressCallback, UpscaleEngine

IS_WINDOWS = sys.platform == "win32"
SETUP_SCRIPT = "scripts/setup.ps1" if IS_WINDOWS else "scripts/setup.sh"

ENGINE_DIR = ENGINES_DIR / "realesrgan"
BINARY_PATH = ENGINE_DIR / ("realesrgan-ncnn-vulkan.exe" if IS_WINDOWS else "realesrgan-ncnn-vulkan")
MODELS_DIR = ENGINE_DIR / "models"


class RealEsrganNcnnEngine(UpscaleEngine):
    def __init__(
        self,
        name: str,
        label: str,
        model_name: str,
        description: str = "",
        supported_scales: tuple[int, ...] = (4,),
    ) -> None:
        self.name = name
        self.label = label
        self.model_name = model_name
        self.description = description
        self.supported_scales = supported_scales

    def is_available(self) -> bool:
        # os.access honours the executable bit on POSIX and is always true
        # for an existing file on Windows, where there is no such bit.
        return BINARY_PATH.is_file() and os.access(BINARY_PATH, os.X_OK)

    def unavailable_reason(self) -> str | None:
        if not ENGINE_DIR.exists():
            return f"not installed — run {SETUP_SCRIPT}"
        if not BINARY_PATH.is_file():
            return f"binary missing at {BINARY_PATH}"
        if not self.is_available():
            return f"{BINARY_PATH} is not executable"
        return None

    def upscale_frames(
        self,
        in_dir: Path,
        out_dir: Path,
        scale: int,
        on_progress: ProgressCallback,
        gpu_id: int | None = None,
    ) -> None:
        total = sum(1 for _ in in_dir.iterdir())
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(BINARY_PATH),
            "-i", str(in_dir),
            "-o", str(out_dir),
            "-n", self.model_name,
            "-s", str(scale),
            "-f", "png",
            "-v",
        ]
        if MODELS_DIR.is_dir():
            cmd += ["-m", str(MODELS_DIR)]
        if gpu_id is not None:
            cmd += ["-g", str(gpu_id)]

        proc = proc_.Popen(
            cmd,
            cwd=ENGINE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        done = 0
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                # -v prints one line per completed frame, e.g. "0001.png done"
                if line.strip().endswith("done"):
                    done += 1
                    on_progress(min(done, total), total)
        except BaseException:
            # on_progress raised (JobCancelled) — don't leave the binary running
            proc.kill()
            proc.wait()
            raise

        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(
                f"realesrgan-ncnn-vulkan exited with code {returncode}"
            )

        on_progress(total, total)
