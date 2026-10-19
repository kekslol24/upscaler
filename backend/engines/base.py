"""Common interface every upscaling engine must implement.

The harness (jobs.py / pipeline.py) only ever talks to engines through
this interface, so adding a new model backend later means writing one
new file here and registering it in engines/__init__.py — nothing else
in the pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..pipeline import RenderOptions, VideoInfo

#: (frames_done, frames_total). May raise JobCancelled, which the engine
#: must let propagate after stopping any child processes it started.
ProgressCallback = Callable[[int, int], None]


class JobCancelled(Exception):
    """Raised from a ProgressCallback when the user cancelled the job."""


class FrameInterpolator(ABC):
    """A frame-rate stage: turns a directory of N frames into a directory
    of `target_count` frames evenly spaced in time from the first to the
    last input frame (so output[0] == input[0] and output[-1] == input[-1]).
    """

    name: str
    label: str
    description: str = ""
    #: True if the interpolator implements interpolate_video() (frames
    #: streamed through pipes, no intermediate PNGs); the harness prefers it.
    streaming: bool = False

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this interpolator's binary/deps are installed and usable."""

    def unavailable_reason(self) -> str | None:
        return None if self.is_available() else "not installed"

    @abstractmethod
    def interpolate_frames(
        self,
        in_dir: Path,
        out_dir: Path,
        target_count: int,
        on_progress: ProgressCallback,
        gpu_id: int | None = None,
    ) -> None:
        """Write `target_count` PNG frames to out_dir, named so that a
        sorted listing is playback order. Must call on_progress with
        (frames_done, target_count) and let JobCancelled propagate after
        stopping any child process."""

    def interpolate_video(
        self,
        input_path: Path,
        output_path: Path,
        info: "VideoInfo",
        factor: int,
        mode: str,
        on_progress: ProgressCallback,
        gpu_id: int | None = None,
        options: "RenderOptions | None" = None,
    ) -> None:
        """Streaming alternative to interpolate_frames(): decode input_path,
        interpolate, and encode straight to output_path without writing
        intermediate frames. Only called when `streaming` is True."""
        raise NotImplementedError


class UpscaleEngine(ABC):
    #: stable machine-readable id, used in the API and job records
    name: str
    #: human-readable label for the frontend dropdown
    label: str
    #: short note shown in the UI (what this model is good at)
    description: str = ""
    #: scale factors this engine supports
    supported_scales: tuple[int, ...] = (2, 4)
    #: True if the engine implements upscale_video() (frames streamed
    #: through pipes, no intermediate PNGs); the harness prefers it.
    streaming: bool = False

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

    def upscale_video(
        self,
        input_path: Path,
        output_path: Path,
        info: "VideoInfo",
        scale: int,
        on_progress: ProgressCallback,
        gpu_id: int | None = None,
        options: "RenderOptions | None" = None,
    ) -> None:
        """Streaming alternative to upscale_frames(): decode input_path,
        upscale, and encode straight to output_path (with the original
        audio) without writing intermediate frames, honouring the clip,
        filters and encoder in `options`. Only called when `streaming` is
        True. frames_total passed to on_progress is an estimate until the
        stream ends.
        """
        raise NotImplementedError
