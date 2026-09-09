"""RIFE frame interpolation on the prebuilt rife-ncnn-vulkan binary
(the same one Flowframes uses), running in directory mode: it reads a
folder of frames and writes `-n` evenly spaced output frames using the
rife-v4.6 model, which supports arbitrary timesteps so any multiplier
works, not just 2x.

scripts/setup.sh / setup.ps1 download the binary and the v4.6 model into
engines_bin/rife/.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from ..paths import ENGINES_DIR
from .base import FrameInterpolator, ProgressCallback

IS_WINDOWS = sys.platform == "win32"
SETUP_SCRIPT = "scripts/setup.ps1" if IS_WINDOWS else "scripts/setup.sh"

ENGINE_DIR = ENGINES_DIR / "rife"
BINARY_PATH = ENGINE_DIR / ("rife-ncnn-vulkan.exe" if IS_WINDOWS else "rife-ncnn-vulkan")


class RifeNcnnInterpolator(FrameInterpolator):
    def __init__(self, name: str, label: str, model_name: str, description: str = "") -> None:
        self.name = name
        self.label = label
        self.model_name = model_name
        self.description = description

    @property
    def model_dir(self) -> Path:
        return ENGINE_DIR / self.model_name

    def is_available(self) -> bool:
        return (
            BINARY_PATH.is_file()
            and os.access(BINARY_PATH, os.X_OK)
            and self.model_dir.is_dir()
        )

    def unavailable_reason(self) -> str | None:
        if not BINARY_PATH.is_file():
            return f"rife-ncnn-vulkan not installed — run {SETUP_SCRIPT}"
        if not os.access(BINARY_PATH, os.X_OK):
            return f"{BINARY_PATH} is not executable"
        if not self.model_dir.is_dir():
            return f"model folder missing: {self.model_dir}"
        return None

    def interpolate_frames(
        self,
        in_dir: Path,
        out_dir: Path,
        target_count: int,
        on_progress: ProgressCallback,
        gpu_id: int | None = None,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(BINARY_PATH),
            "-i", str(in_dir),
            "-o", str(out_dir),
            "-n", str(target_count),
            "-m", str(self.model_dir),
            "-f", "%08d.png",
        ]
        if gpu_id is not None:
            cmd += ["-g", str(gpu_id)]

        proc = subprocess.Popen(
            cmd, cwd=ENGINE_DIR,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        # The binary's console output isn't a reliable per-frame signal, so
        # progress is the number of frames it has written so far.
        try:
            while proc.poll() is None:
                on_progress(min(_count_pngs(out_dir), target_count), target_count)
                time.sleep(0.5)
        except BaseException:
            proc.kill()
            proc.wait()
            raise

        if proc.returncode != 0:
            err = (proc.stderr.read() if proc.stderr else "").strip()
            raise RuntimeError(f"rife-ncnn-vulkan exited with code {proc.returncode}: {err}")
        written = _count_pngs(out_dir)
        if written != target_count:
            raise RuntimeError(f"rife-ncnn-vulkan wrote {written} frames, expected {target_count}")
        on_progress(target_count, target_count)


def _count_pngs(directory: Path) -> int:
    try:
        return sum(1 for p in os.scandir(directory) if p.name.endswith(".png"))
    except FileNotFoundError:
        return 0
