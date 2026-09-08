"""Common interface every upscaling engine must implement.

The harness (jobs.py / pipeline.py) only ever talks to engines through
this interface, so adding a new model backend later means writing one
new file here and registering it in engines/__init__.py — nothing else
in the pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

ProgressCallback = Callable[[int, int], None]  # (frames_done, frames_total)


class UpscaleEngine(ABC):
    #: stable machine-readable id, used in the API and job records
    name: str
    #: human-readable label for the frontend dropdown
    label: str
    #: short note shown in the UI (what this model is good at)
    description: str = ""
    #: scale factors this engine supports
    supported_scales: tuple[int, ...] = (2, 4)

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this engine's binary/deps are installed and usable."""

    def unavailable_reason(self) -> str | None:
        """Human-readable reason is_available() is False, or None."""
        return None if self.is_available() else "not installed"

    @abstractmethod
    def upscale_frames(
        self,
        in_dir: Path,
        out_dir: Path,
        scale: int,
        on_progress: ProgressCallback,
        gpu_id: int | None = None,
    ) -> None:
        """Upscale every frame image in in_dir, writing results to out_dir.

        Output filenames must match input filenames so the reassembly step
        can pick the original frame rate back up. Must call on_progress
        periodically with (frames_done, frames_total). gpu_id selects a
        specific Vulkan device (index from gpu.list_gpus()); None lets the
        engine pick its own default.
        """
