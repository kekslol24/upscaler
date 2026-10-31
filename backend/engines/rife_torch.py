"""RIFE v4.6 frame interpolation through PyTorch on CUDA.

Unlike the ncnn engine this path never writes PNG frames: decoded RGB
frames go straight into the model and the generated frames go straight
back into ffmpeg. At 4K that is the difference between disk-bound
interpolation and running the network at full GPU speed.
"""

from __future__ import annotations

import importlib.util
import os
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from .. import pipeline
from ..paths import MODELS_DIR
from .base import FrameInterpolator, ProgressCallback

if TYPE_CHECKING:
    from .rife_torch_model import RifeModel

MODEL_DIR = MODELS_DIR / "rife-v4.6"
WEIGHTS_PATH = MODEL_DIR / "flownet.pkl"

#: above 1080p-ish, run RIFE's half-resolution flow mode
_UHD_THRESHOLD = 1920 * 1080 * 1.5
_MODELS: dict[tuple[str, str], "RifeModel"] = {}

#: Reusable pinned host buffers for the GPU->CPU frame download.
#: A blocking `.cpu()` of one 4K RGB frame measured ~1.5 GB/s (15-40 ms) on
#: the RunPod 4090, while a pinned copy runs at ~20 GB/s (~1-2 ms), and the
#: download was the dominant per-frame cost. Reuse one buffer per frame
#: shape instead of allocating (and page-faulting) a fresh pageable one.
_PINNED_HOST: dict[tuple[int, ...], "torch.Tensor"] = {}


def _pinned_host(torch, shape: tuple[int, ...]):
    buf = _PINNED_HOST.get(shape)
    if buf is None:
        try:
            buf = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
        except Exception:
            # Some ROCm/Windows builds cannot allocate pinned memory; the
            # copy still works, it just becomes synchronous.
            buf = torch.empty(shape, dtype=torch.uint8)
        _PINNED_HOST[shape] = buf
    return buf


def _cuda_visible_devices() -> str | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    return raw or "(empty)"


def _cuda_state() -> dict:
    """Inspect the CUDA situation without raising.

    Returns one of:
      {"ok": bool, "error": str|None, "device_count": int|None,
       "visible": str|None, "available": bool}
    `available=False` means torch is importable but CUDA is not usable
    (the whole reason this stack falls back to Vulkan/CPU engines).
    """
    if importlib.util.find_spec("torch") is None:
        return {"ok": False, "error": "PyTorch is not installed", "available": False, "device_count": None, "visible": _cuda_visible_devices()}
    try:
        import torch
    except Exception as exc:
        return {"ok": False, "error": f"PyTorch failed to import: {exc}", "available": False, "device_count": None, "visible": _cuda_visible_devices()}
    if not torch.cuda.is_available():
        return {"ok": False, "error": None, "available": False, "device_count": 0, "visible": _cuda_visible_devices()}
    count = torch.cuda.device_count()
    if count == 0:
        return {"ok": False, "error": "PyTorch reports zero CUDA devices despite CUDA being available", "available": True, "device_count": 0, "visible": _cuda_visible_devices()}
    return {"ok": True, "error": None, "available": True, "device_count": count, "visible": _cuda_visible_devices()}


class _AsyncFrameWriter:
    """Feed ffmpeg's stdin from a background thread.

    The encoder runs on the CPU while the next RIFE forward pass runs on
    the GPU, instead of the two alternating. Frames are copied to `bytes`
    before queueing because the producer reuses one pinned host buffer.
    """

    def __init__(self, writer, depth: int = 4) -> None:
        self._writer = writer
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                data = self._queue.get()
                if data is None:
                    return
                self._writer.stdin.write(data)
        except BaseException as exc:  # broken pipe if ffmpeg dies
            self._error = exc

    def write(self, data) -> None:
        if self._error is not None:
            raise self._error
        payload = bytes(data)
        while True:
            try:
                self._queue.put(payload, timeout=0.5)
                return
            except queue.Full:
                # Never block forever if the encoder thread has died.
                if self._error is not None:
                    raise self._error

    def close(self) -> None:
        # Deliver the sentinel only while the consumer is alive: if ffmpeg
        # died the queue may be full and an unconditional put() would hang.
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=0.5)
                break
            except queue.Full:
                if self._error is not None:
                    break
        self._thread.join()
        if self._error is not None:
            raise self._error

    def abort(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5)


class RifeTorchInterpolator(FrameInterpolator):
    streaming = True

    def __init__(self, name: str, label: str, description: str = "") -> None:
        self.name = name
        self.label = label
        self.description = description

    @property
    def model_path(self) -> Path:
        return WEIGHTS_PATH

    def is_available(self) -> bool:
        if not self.model_path.is_file():
            return False
        return _cuda_state()["ok"]

    def unavailable_reason(self) -> str | None:
        if not self.model_path.is_file():
            return f"RIFE PyTorch weights missing: {self.model_path}"
        state = _cuda_state()
        # A specific reason (not installed / import failed / zero devices)
        # wins over the generic "CUDA not available" hint.
        if state["error"]:
            return state["error"]
        if not state["available"]:
            raw = state["visible"]
            hint = (
                f" (CUDA_VISIBLE_DEVICES={raw!r})"
                if raw not in (None, "", "-1")
                else ""
            )
            return (
                "PyTorch CUDA is not available"
                f"{hint}; nvidia-smi may show GPUs that torch cannot see — "
                "check CUDA_VISIBLE_DEVICES / the container driver permissions"
            )
        return None

    def interpolate_video(
        self,
        input_path: Path,
        output_path: Path,
        info: pipeline.VideoInfo,
        factor: int,
        mode: str,
        on_progress: ProgressCallback,
        gpu_id: int | None = None,
        options: pipeline.RenderOptions | None = None,
    ) -> None:
        options = options or pipeline.RenderOptions()
        factor = max(1, int(factor))
        torch, model = self._load(gpu_id)
        scale = self._scale(info.width, info.height)
        if mode == "smooth":
            out_rate = pipeline.scale_frame_rate(info.frame_rate, factor)
            audio_source = input_path if info.has_audio else None
        elif mode == "slowmo":
            out_rate = info.frame_rate
            audio_source = None
        else:
            raise ValueError(f"unknown interpolation mode: {mode!r}")

        frame_bytes = info.width * info.height * 3
        total = max(info.frames_in(options.clip), 1)
        # yuv420p needs even dimensions. Converting RGB->YUV in ffmpeg costs
        # ~30 ms per 4K frame and caps the stream at ~20 fps, so do it on the
        # GPU and hand ffmpeg YUV directly (half the pipe payload too).
        use_yuv = info.width % 2 == 0 and info.height % 2 == 0
        to_bytes = self._to_yuv_bytes if use_yuv else self._to_bytes
        reader = pipeline.open_frame_reader(input_path, options)
        writer = None
        sink = None
        done = 0
        try:
            first = _read_exact(reader.stdout, frame_bytes)
            if first is None:
                raise RuntimeError("no frames decoded from the source")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pipeline.open_frame_writer(
                output_path, info.width, info.height, out_rate, audio_source, options,
                pix_fmt="yuv420p" if use_yuv else "rgb24",
            )
            sink = _AsyncFrameWriter(writer)
            prev = self._to_tensor(torch, first, info.width, info.height, model.device)
            sink.write(to_bytes(torch, prev))
            done = 1
            on_progress(done, total)

            while True:
                buf = _read_exact(reader.stdout, frame_bytes)
                if buf is None:
                    break
                cur = self._to_tensor(torch, buf, info.width, info.height, model.device)
                for step in range(1, factor):
                    out = self._infer(model, prev, cur, step / factor, scale)
                    sink.write(to_bytes(torch, out))
                sink.write(to_bytes(torch, cur))
                prev = cur
                done += 1
                on_progress(done, total)
            sink.close()
            sink = None
            self._finish_writer(writer)
            writer = None
            _check_reader(reader)
            reader = None
            on_progress(done, total)
        finally:
            if sink is not None:
                sink.abort()
            if writer is not None:
                _terminate(writer)
            if reader is not None:
                _terminate(reader)

    def interpolate_frames(
        self,
        in_dir: Path,
        out_dir: Path,
        target_count: int,
        on_progress: ProgressCallback,
        gpu_id: int | None = None,
    ) -> None:
        torch, model = self._load(gpu_id)
        files = sorted(p for p in in_dir.iterdir() if p.suffix == ".png")
        if not files:
            raise RuntimeError(f"no PNG frames in {in_dir}")
        if target_count <= 0:
            return
        width, height = _png_size(files[0])
        if not width or not height:
            raise RuntimeError(f"cannot read PNG dimensions: {files[0]}")
        scale = self._scale(width, height)
        frame_bytes = width * height * 3
        out_dir.mkdir(parents=True, exist_ok=True)
        reader = pipeline.open_frame_reader(in_dir / "%08d.png")
        writer = pipeline.open_frame_writer(out_dir / "%08d.png", width, height, "1")
        try:
            first = _read_exact(reader.stdout, frame_bytes)
            if first is None:
                raise RuntimeError(f"no frames decoded from {in_dir}")
            cur_bytes = first
            cur = self._to_tensor(torch, first, width, height, model.device)
            cur_idx = 0
            nxt_bytes = None
            nxt = None
            span = len(files) - 1
            for out_idx in range(target_count):
                pos = out_idx * span / max(target_count - 1, 1)
                src_idx = min(span, int(pos))
                frac = pos - src_idx
                while cur_idx < src_idx:
                    if nxt_bytes is None:
                        nxt_bytes = _read_exact(reader.stdout, frame_bytes)
                        if nxt_bytes is None:
                            raise RuntimeError(f"frame stream ended early in {in_dir}")
                        nxt = self._to_tensor(torch, nxt_bytes, width, height, model.device)
                    cur_bytes, cur = nxt_bytes, nxt
                    nxt_bytes, nxt = None, None
                    cur_idx += 1
                if frac <= 1e-6 or src_idx >= span:
                    writer.stdin.write(cur_bytes)
                else:
                    if nxt_bytes is None:
                        nxt_bytes = _read_exact(reader.stdout, frame_bytes)
                        if nxt_bytes is None:
                            raise RuntimeError(f"frame stream ended early in {in_dir}")
                        nxt = self._to_tensor(torch, nxt_bytes, width, height, model.device)
                    out = self._infer(model, cur, nxt, frac, scale)
                    writer.stdin.write(self._to_bytes(torch, out))
                on_progress(out_idx + 1, target_count)
            self._finish_writer(writer)
            writer = None
            on_progress(target_count, target_count)
        finally:
            if writer is not None:
                _terminate(writer)
            _terminate(reader)

    def _load(self, gpu_id: int | None):
        import torch
        from .rife_torch_model import RifeModel

        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch CUDA is not available"
            )
        count = torch.cuda.device_count()
        if gpu_id is not None and not (0 <= gpu_id < count):
            raise RuntimeError(
                f"no usable CUDA device with index {gpu_id}: "
                f"torch sees {count} CUDA device(s) "
                f"(CUDA_VISIBLE_DEVICES={_cuda_visible_devices()!r})"
            )
        index = gpu_id if gpu_id is not None else 0
        device = torch.device(f"cuda:{index}")
        key = (str(device), str(self.model_path))
        model = _MODELS.get(key)
        if model is None:
            torch.cuda.set_device(device)
            torch.backends.cudnn.benchmark = True
            model = RifeModel(self.model_path, device)
            _MODELS[key] = model
        return torch, model

    @staticmethod
    def _scale(width: int, height: int) -> float:
        return 0.5 if width * height > _UHD_THRESHOLD else 1.0

    @staticmethod
    def _infer(model: "RifeModel", prev, cur, timestep: float,
               scale: float):
        import torch.nn.functional as F

        height, width = prev.shape[-2:]
        block = max(128, int(128 / scale))
        pad_h = (block - height % block) % block
        pad_w = (block - width % block) % block
        if pad_h or pad_w:
            prev = F.pad(prev, (0, pad_w, 0, pad_h))
            cur = F.pad(cur, (0, pad_w, 0, pad_h))
        out = model.infer(prev, cur, timestep, scale)
        return out[..., :height, :width]

    @staticmethod
    def _to_tensor(torch, buf: bytes, width: int, height: int, device):
        frame = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
        frame = frame.view(height, width, 3).permute(2, 0, 1).unsqueeze(0)
        return frame.to(device=device, non_blocking=True).float().div_(255.0)

    @staticmethod
    def _to_bytes(torch, frame) -> "memoryview | bytes":
        frame = frame[0].clamp(0.0, 1.0).mul(255.0).add(0.5)
        frame = frame.clamp(0.0, 255.0).to(torch.uint8)
        frame = frame.permute(1, 2, 0).contiguous()
        if frame.device.type != "cuda":
            return frame.cpu().numpy().tobytes()
        # Copy into a pinned buffer asynchronously; the caller writes it to
        # ffmpeg's stdin before the next frame overwrites the buffer.
        host = _pinned_host(torch, tuple(frame.shape))
        host.copy_(frame, non_blocking=True)
        torch.cuda.synchronize()
        return memoryview(host.numpy())

    @staticmethod
    def _to_yuv_bytes(torch, frame) -> "memoryview | bytes":
        """BT.601 limited-range yuv420p, matching ffmpeg's default
        rgb24 -> yuv420p conversion (measured MAE < 1 level)."""
        from torch.nn.functional import avg_pool2d

        r, g, b = frame[0, 0], frame[0, 1], frame[0, 2]
        y = 16.0 + 219.0 * (0.299 * r + 0.587 * g + 0.114 * b)
        u = 128.0 + 224.0 * (-0.168736 * r - 0.331264 * g + 0.5 * b)
        v = 128.0 + 224.0 * (0.5 * r - 0.418688 * g - 0.081312 * b)
        u = avg_pool2d(u[None, None], 2)[0, 0]
        v = avg_pool2d(v[None, None], 2)[0, 0]
        planes = torch.cat([
            y.clamp(0.0, 255.0).round().to(torch.uint8).reshape(-1),
            u.clamp(0.0, 255.0).round().to(torch.uint8).reshape(-1),
            v.clamp(0.0, 255.0).round().to(torch.uint8).reshape(-1),
        ])
        if planes.device.type != "cuda":
            return planes.cpu().numpy().tobytes()
        host = _pinned_host(torch, tuple(planes.shape))
        host.copy_(planes, non_blocking=True)
        torch.cuda.synchronize()
        return memoryview(host.numpy())

    @staticmethod
    def _finish_writer(writer) -> None:
        try:
            writer.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        err = writer.stderr.read().decode(errors="replace").strip()
        code = writer.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg encode failed ({code}): {err}")


def _read_exact(stream, size: int) -> bytes | None:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == size:
                return None
            raise RuntimeError("unexpected end of decoded frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _check_reader(reader) -> None:
    try:
        reader.stdout.close()
    except OSError:
        pass
    err = reader.stderr.read().decode(errors="replace").strip()
    code = reader.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg decode failed ({code}): {err}")


def _terminate(proc) -> None:
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    try:
        if proc.stdout is not None:
            proc.stdout.close()
    except OSError:
        pass
    if proc.poll() is None:
        proc.kill()
    proc.wait()


def _png_size(path: Path) -> tuple[int, int]:
    try:
        with open(path, "rb") as f:
            header = f.read(24)
    except OSError:
        return 0, 0
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return 0, 0
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
