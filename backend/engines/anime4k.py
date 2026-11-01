"""Anime4K v4 upscaling through ffmpeg's libplacebo filter.

The bundled GLSL shaders are the official Anime4K v4 chain (Mode A HQ):
highlight clamp, Restore CNN VL, Upscale CNN x2 VL, the two
AutoDownscalePre passes, and Upscale CNN x2 M. libplacebo runs the whole
chain on the Vulkan GPU in one ffmpeg pass, so decode, shader upscaling
and encode overlap and nothing is written to disk in between.

On WSL the Linux ffmpeg's libplacebo goes through Mesa's Dozen
(Vulkan-on-D3D12) and is very slow for 4K frames. If a Windows ffmpeg
with libplacebo is installed, this engine runs that instead: it uses the
native AMD Vulkan driver and can optionally encode with AMF. Paths are
translated with wslpath; on native Linux/Windows the normal ffmpeg is
used.
"""

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path

from .. import pipeline, proc
from ..paths import DATA_DIR, MODELS_DIR
from .base import ProgressCallback, UpscaleEngine

SHADER_DIR = MODELS_DIR / "anime4k" / "shaders"
#: official Anime4K Mode A (HQ), in mpv's `~~/shaders/...` order
SHADER_CHAIN = (
    "Anime4K_Clamp_Highlights.glsl",
    "Anime4K_Restore_CNN_VL.glsl",
    "Anime4K_Upscale_CNN_x2_VL.glsl",
    "Anime4K_AutoDownscalePre_x2.glsl",
    "Anime4K_AutoDownscalePre_x4.glsl",
    "Anime4K_Upscale_CNN_x2_M.glsl",
)
_CACHE_DIR = DATA_DIR / "anime4k"
_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class _Backend:
    ffmpeg: str
    #: True for a Windows ffmpeg.exe called from WSL, where paths must be
    #: translated from /home/... to \\wsl.localhost\... (or C:\...).
    windows_interop: bool


def _is_wsl() -> bool:
    if sys.platform == "win32":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text(errors="ignore").lower()
    except OSError:
        return False


def _has_libplacebo(binary: str) -> bool:
    try:
        out = proc.run(
            [binary, "-hide_banner", "-filters"],
            capture_output=True, text=True, errors="replace",
            timeout=20, check=True,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return " libplacebo " in out


def _windows_ffmpeg() -> str | None:
    if not _is_wsl():
        return None
    cmd = shutil.which("cmd.exe") or "/mnt/c/Windows/System32/cmd.exe"
    if not os.path.isfile(cmd):
        return None
    try:
        out = proc.run(
            [cmd, "/c", "where", "ffmpeg"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, errors="replace", timeout=20,
        ).stdout
    except (subprocess.TimeoutExpired, OSError):
        return None
    for line in out.splitlines():
        win_path = line.strip()
        if not win_path:
            continue
        try:
            host_path = proc.run(
                ["wslpath", "-u", win_path],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            continue
        if host_path and os.path.isfile(host_path):
            return host_path
    return None


@functools.lru_cache(maxsize=1)
def _backend() -> _Backend | None:
    # Prefer the native Windows ffmpeg under WSL; Dozen's Vulkan path is
    # several times slower for these shaders.
    if _is_wsl():
        binary = _windows_ffmpeg()
        if binary and _has_libplacebo(binary):
            return _Backend(binary, windows_interop=True)
    binary = shutil.which("ffmpeg")
    if binary and _has_libplacebo(binary):
        return _Backend(binary, windows_interop=False)
    return None


def shader_chain_path() -> Path:
    """Concatenate the shader files into the single .hook file libplacebo
    accepts. It is content-addressed, so editing a shader creates a new
    cache file instead of serving a stale chain."""
    content = b"".join(
        (SHADER_DIR / name).read_bytes() + b"\n" for name in SHADER_CHAIN
    )
    digest = hashlib.sha256(content).hexdigest()[:16]
    target = _CACHE_DIR / f"anime4k-mode-a-hq-{digest}.hook"
    if target.is_file():
        return target
    with _CACHE_LOCK:
        if not target.is_file():
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f"{target.name}.{threading.get_ident()}.tmp")
            tmp.write_bytes(content)
            tmp.replace(target)
    return target


def _host_path(path: Path, backend: _Backend) -> str:
    """Translate a WSL path for a Windows ffmpeg (e.g. to a UNC path)."""
    if not backend.windows_interop:
        return str(path)
    try:
        return proc.run(
            ["wslpath", "-w", str(path)],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return str(path)


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"cannot read PNG dimensions: {path}")
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def _libplacebo_filter(
    width: int, height: int, shader: Path, pix_fmt: str
) -> str:
    # The shader filename is relative to the ffmpeg cwd (the cache dir),
    # which avoids escaping drive colons inside the filter graph.
    return (
        f"libplacebo=w={width}:h={height}:"
        f"custom_shader_path={shader.name}:format={pix_fmt}"
    )


@functools.lru_cache(maxsize=8)
def _encoder_available(binary: str, name: str) -> bool:
    try:
        out = proc.run(
            [binary, "-hide_banner", "-encoders"],
            capture_output=True, text=True, errors="replace",
            timeout=20, check=True,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return f" {name} " in out


def _encode_args(
    settings: pipeline.EncodeSettings, container: str, backend: _Backend
) -> list[str]:
    # A Windows ffmpeg reached from WSL can use the hardware AMF encoder,
    # which the Linux ffmpeg's encoder probe does not know about.
    if backend.windows_interop and settings.encoder == "auto":
        name = "hevc_amf" if settings.codec == "hevc" else "h264_amf"
        if _encoder_available(backend.ffmpeg, name):
            settings = replace(settings, encoder=name)
    return pipeline.encode_args(settings, container)


class Anime4KEngine(UpscaleEngine):
    streaming = True
    #: The actual work happens in a child ffmpeg process. The cooperative
    #: pause used by the in-process pipelines would only park this Python
    #: thread (and, under WSL, the Windows interop wrapper), not ffmpeg.
    pausable = False

    def __init__(
        self,
        name: str,
        label: str,
        description: str = "",
        supported_scales: tuple[int, ...] = (2, 3, 4),
    ) -> None:
        self.name = name
        self.label = label
        self.description = description
        self.supported_scales = supported_scales

    # -- availability -----------------------------------------------------

    def is_available(self) -> bool:
        return self.unavailable_reason() is None

    def unavailable_reason(self) -> str | None:
        missing = [name for name in SHADER_CHAIN if not (SHADER_DIR / name).is_file()]
        if missing:
            return f"Anime4K shader missing: {missing[0]}"
        if _backend() is None:
            return (
                "no ffmpeg with libplacebo found — install an ffmpeg built "
                "with --enable-libplacebo"
            )
        return None

    # -- ffmpeg runner ----------------------------------------------------

    def _run(
        self,
        input_args: list[str],
        output_args: list[str],
        filters: list[str],
        shader: Path,
        total: int,
        on_progress: ProgressCallback,
        audio: bool = False,
    ) -> None:
        backend = _backend()
        if backend is None:
            raise RuntimeError(self.unavailable_reason() or "Anime4K unavailable")
        cmd = [
            backend.ffmpeg, "-y", "-v", "error", "-nostdin",
            "-progress", "pipe:1", "-nostats",
            *input_args,
            "-map", "0:v:0",
        ]
        if audio:
            cmd += ["-map", "0:a?"]
        cmd += ["-vf", ",".join(filters), *output_args]
        process = proc.Popen(
            cmd,
            cwd=str(shader.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        errors: list[str] = []

        def drain_stderr() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                errors.append(line)

        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()
        done = 0
        try:
            assert process.stdout is not None
            for line in process.stdout:
                if line.startswith("frame="):
                    try:
                        done = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        continue
                    if done:
                        on_progress(min(done, total), total)
            process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            stderr_thread.join(timeout=2)
        if process.returncode != 0:
            err = "".join(errors).strip()
            last = err.splitlines()[-1] if err else "no error output"
            raise RuntimeError(f"Anime4K render failed ({process.returncode}): {last}")
        on_progress(total, total)

    # -- UpscaleEngine ----------------------------------------------------

    def upscale_video(
        self,
        input_path: Path,
        output_path: Path,
        info: pipeline.VideoInfo,
        scale: int,
        on_progress: ProgressCallback,
        gpu_id: int | None = None,
        options: pipeline.RenderOptions | None = None,
    ) -> None:
        options = options or pipeline.RenderOptions()
        backend = _backend()
        if backend is None:
            raise RuntimeError(self.unavailable_reason() or "Anime4K unavailable")
        shader = shader_chain_path()
        libplacebo = _libplacebo_filter(
            info.width * scale, info.height * scale, shader, "yuv420p"
        )
        filters = [*options.filters.pre(), libplacebo, *options.filters.post()]
        self._run(
            [*pipeline._clip_input_args(options.clip),
             "-i", _host_path(input_path, backend)],
            [
                *_encode_args(options.encode, output_path.suffix.lstrip("."), backend),
                _host_path(output_path, backend),
            ],
            filters,
            shader,
            max(info.frames_in(options.clip), 1),
            on_progress,
            audio=info.has_audio,
        )

    def upscale_frames(
        self,
        in_dir: Path,
        out_dir: Path,
        scale: int,
        on_progress: ProgressCallback,
        gpu_id: int | None = None,
    ) -> None:
        backend = _backend()
        if backend is None:
            raise RuntimeError(self.unavailable_reason() or "Anime4K unavailable")
        frames = sorted(p for p in in_dir.iterdir() if p.suffix == ".png")
        if not frames:
            raise RuntimeError(f"no PNG frames in {in_dir}")
        width, height = _png_size(frames[0])
        out_dir.mkdir(parents=True, exist_ok=True)
        shader = shader_chain_path()
        self._run(
            ["-framerate", "1", "-start_number", "0",
             "-i", _host_path(in_dir / "%08d.png", backend)],
            ["-start_number", "0", "-compression_level", "1",
             _host_path(out_dir / "%08d.png", backend)],
            [_libplacebo_filter(width * scale, height * scale, shader, "rgb24")],
            shader,
            len(frames),
            on_progress,
        )

    def upscale_frames_to_video(
        self,
        in_dir: Path,
        output_path: Path,
        frame_rate: str,
        scale: int,
        on_progress: ProgressCallback,
        gpu_id: int | None = None,
        options: pipeline.RenderOptions | None = None,
    ) -> None:
        options = options or pipeline.RenderOptions()
        backend = _backend()
        if backend is None:
            raise RuntimeError(self.unavailable_reason() or "Anime4K unavailable")
        frames = sorted(p for p in in_dir.iterdir() if p.suffix == ".png")
        if not frames:
            raise RuntimeError(f"no PNG frames in {in_dir}")
        width, height = _png_size(frames[0])
        shader = shader_chain_path()
        libplacebo = _libplacebo_filter(
            width * scale, height * scale, shader, "yuv420p"
        )
        self._run(
            ["-framerate", frame_rate, "-start_number", "0",
             "-i", _host_path(in_dir / "%08d.png", backend)],
            [
                *_encode_args(options.encode, output_path.suffix.lstrip("."), backend),
                _host_path(output_path, backend),
            ],
            [libplacebo, *options.filters.post()],
            shader,
            len(frames),
            on_progress,
        )
