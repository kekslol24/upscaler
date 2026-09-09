"""Real-ESRGAN (realesr-animevideov3) on ONNX Runtime with a GPU provider.

Same weights as the ncnn engine: models/*.onnx are exported from the
official .pth by scripts/export_onnx.py, and the output matches ncnn's to
within one 8-bit level. It is several times faster on the same GPU
because it runs each frame in a few full-width bands instead of 400px
tiles (whole-frame passes trip a DirectML large-tensor bug — see
_MAX_INTERMEDIATE_ELEMENTS), and
because it streams: ffmpeg decodes straight into the model and upscaled
frames go straight into the encoder. Nothing is written to disk in
between, so decode, upscale and encode overlap and a job never needs
tens of GB of intermediate PNGs.

Runtime: onnxruntime-directml on Windows (any DX12 GPU: AMD, NVIDIA,
Intel). On Linux the same models work with onnxruntime-gpu (CUDA/ROCm)
if installed. The CPU provider is refused as far too slow.

gpu_id is passed through as the provider's device_id. DirectML numbers
adapters in DXGI order, which on a single-GPU machine is 0 like Vulkan;
on multi-GPU systems the two orders can differ.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
from pathlib import Path

from .. import pipeline
from ..paths import MODELS_DIR
from ..pipeline import RenderOptions, VideoInfo
from .base import JobCancelled, ProgressCallback, UpscaleEngine
_GPU_PROVIDERS = ("DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider")
#: set UPSCALER_ALLOW_CPU=1 to run on the CPU provider (tests, GPU-less
#: machines) — roughly 20 s per 1080p frame, so refused by default
_ALLOW_CPU_ENV = "UPSCALER_ALLOW_CPU"
_QUEUE_DEPTH = 4

# DirectML silently zero-fills the tail of very large op outputs (a 1080p
# frame's x4 intermediate, ~100M elements, comes back with the bottom half
# black), so frames are processed in full-width horizontal bands sized to
# keep the biggest intermediate — 3 channels x 4H x 4W after PixelShuffle —
# under this element count. Bands overlap and the overlap is cropped away,
# so band output is identical to whole-frame output.
_MAX_INTERMEDIATE_ELEMENTS = int(os.environ.get("UPSCALER_BAND_ELEMENTS", 32 * 1024 * 1024))
_BAND_OVERLAP = 16


def _max_band_rows(width: int) -> int:
    return max(32, _MAX_INTERMEDIATE_ELEMENTS // (3 * 4 * 4 * width))


def _infer_frame(session, input_name: str, frame, scale: int):
    """Run one (1, H, W, 3) uint8 frame through the model, splitting it
    into overlapping horizontal bands if it is too tall for one pass."""
    import numpy as np

    height, width = frame.shape[1], frame.shape[2]
    band_rows = _max_band_rows(width)
    if height <= band_rows:
        return session.run(None, {input_name: frame})[0]

    out = np.empty((1, height * scale, width * scale, 3), dtype=np.uint8)
    for start in range(0, height, band_rows):
        end = min(start + band_rows, height)
        pad_top = min(_BAND_OVERLAP, start)
        pad_bottom = min(_BAND_OVERLAP, height - end)
        band = np.ascontiguousarray(frame[:, start - pad_top : end + pad_bottom])
        band_out = session.run(None, {input_name: band})[0]
        out[:, start * scale : end * scale] = band_out[
            :, pad_top * scale : band_out.shape[1] - pad_bottom * scale
        ]
    return out


def _onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    return ort


class RealEsrganOnnxEngine(UpscaleEngine):
    streaming = True

    def __init__(
        self,
        name: str,
        label: str,
        model_prefix: str,
        description: str = "",
        supported_scales: tuple[int, ...] = (2, 3, 4),
    ) -> None:
        self.name = name
        self.label = label
        self.model_prefix = model_prefix
        self.description = description
        self.supported_scales = supported_scales
        self._sessions: dict[tuple[int, int | None], object] = {}
        self._lock = threading.Lock()

    # -- availability -----------------------------------------------------

    def _model_path(self, scale: int) -> Path:
        return MODELS_DIR / f"{self.model_prefix}-x{scale}.onnx"

    def _gpu_provider(self) -> str | None:
        ort = _onnxruntime()
        if ort is None:
            return None
        available = ort.get_available_providers()
        provider = next((p for p in _GPU_PROVIDERS if p in available), None)
        if provider is None and os.environ.get(_ALLOW_CPU_ENV) == "1":
            provider = "CPUExecutionProvider"
        return provider

    def is_available(self) -> bool:
        return self.unavailable_reason() is None

    def unavailable_reason(self) -> str | None:
        if _onnxruntime() is None:
            return "onnxruntime not installed — pip install -r requirements.txt"
        if self._gpu_provider() is None:
            return ("onnxruntime has no GPU provider — needs onnxruntime-directml "
                    f"(Windows) or onnxruntime-gpu (Linux); set {_ALLOW_CPU_ENV}=1 "
                    "to run (very slowly) on the CPU")
        for scale in self.supported_scales:
            if not self._model_path(scale).is_file():
                return f"model file missing: {self._model_path(scale)}"
        return None

    def _session(self, scale: int, gpu_id: int | None):
        ort = _onnxruntime()
        key = (scale, gpu_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                name = self._gpu_provider()
                device = {} if gpu_id is None or name == "CPUExecutionProvider" else {"device_id": gpu_id}
                session = ort.InferenceSession(str(self._model_path(scale)), options, providers=[(name, device)])
                self._sessions[key] = session
            return session

    # -- streaming core ---------------------------------------------------

    def _run_stream(
        self,
        session,
        reader: subprocess.Popen,
        writer: subprocess.Popen,
        width: int,
        height: int,
        scale: int,
        total: int,
        on_progress: ProgressCallback,
    ) -> int:
        """Pump raw RGB frames reader -> model -> writer. Returns frames done."""
        import numpy as np

        frame_bytes = width * height * 3
        input_name = session.get_inputs()[0].name
        in_q: "queue.Queue[bytes | None]" = queue.Queue(maxsize=_QUEUE_DEPTH)
        out_q: "queue.Queue[bytes | None]" = queue.Queue(maxsize=_QUEUE_DEPTH)
        failed = threading.Event()
        errors: list[BaseException] = []

        def read_loop() -> None:
            try:
                assert reader.stdout is not None
                while not failed.is_set():
                    buf = reader.stdout.read(frame_bytes)
                    if len(buf) < frame_bytes:
                        break
                    in_q.put(buf)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                failed.set()
            finally:
                in_q.put(None)

        def write_loop() -> None:
            try:
                assert writer.stdin is not None
                while True:
                    buf = out_q.get()
                    if buf is None:
                        break
                    if not failed.is_set():
                        writer.stdin.write(buf)
                writer.stdin.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                failed.set()
                # keep draining so the main loop never blocks on a full queue
                while out_q.get() is not None:
                    pass

        threads = [threading.Thread(target=read_loop, daemon=True),
                   threading.Thread(target=write_loop, daemon=True)]
        for t in threads:
            t.start()

        done = 0
        try:
            while True:
                buf = in_q.get()
                if buf is None or failed.is_set():
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(1, height, width, 3)
                out = _infer_frame(session, input_name, frame, scale)
                out_q.put(out.tobytes())
                done += 1
                on_progress(done, max(total, done))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            failed.set()
        finally:
            out_q.put(None)
            if failed.is_set():
                reader.kill()
                writer.kill()
                # unblock the reader if it is waiting on a full in_q
                while not in_q.empty():
                    in_q.get_nowait()
            for t in threads:
                t.join()

        read_err = (reader.stderr.read() if reader.stderr else b"").decode(errors="replace").strip()
        write_err = (writer.stderr.read() if writer.stderr else b"").decode(errors="replace").strip()
        reader.wait()
        writer.wait()
        if errors:
            if isinstance(errors[0], JobCancelled):
                raise errors[0]
            raise RuntimeError(f"upscale stream failed: {errors[0]}") from errors[0]
        if reader.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed ({reader.returncode}): {read_err}")
        if writer.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed ({writer.returncode}): {write_err}")
        if done == 0:
            raise RuntimeError(f"no frames decoded: {read_err}")
        return done

    # -- UpscaleEngine ----------------------------------------------------

    def upscale_video(
        self,
        input_path: Path,
        output_path: Path,
        info: VideoInfo,
        scale: int,
        on_progress: ProgressCallback,
        gpu_id: int | None = None,
        options: RenderOptions | None = None,
    ) -> None:
        options = options or RenderOptions()
        session = self._session(scale, gpu_id)
        reader = pipeline.open_frame_reader(input_path, options)
        writer = pipeline.open_frame_writer(
            output_path, info.width * scale, info.height * scale, info.frame_rate,
            audio_source=input_path if info.has_audio else None,
            options=options,
        )
        done = self._run_stream(session, reader, writer, info.width, info.height,
                                scale, info.frames_in(options.clip), on_progress)
        on_progress(done, done)

    def upscale_frames(
        self,
        in_dir: Path,
        out_dir: Path,
        scale: int,
        on_progress: ProgressCallback,
        gpu_id: int | None = None,
    ) -> None:
        """Directory mode, same streaming core over PNG sequences."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self._upscale_sequence(in_dir, out_dir / "%08d.png", "1", scale, on_progress, gpu_id)

    def upscale_frames_to_video(
        self,
        in_dir: Path,
        output_path: Path,
        frame_rate: str,
        scale: int,
        on_progress: ProgressCallback,
        gpu_id: int | None = None,
        options: RenderOptions | None = None,
    ) -> None:
        """Directory in, encoded video out (no audio): lets the chunked
        pipeline skip writing and re-reading upscaled 4K PNGs when this
        engine is the last stage."""
        self._upscale_sequence(in_dir, output_path, frame_rate, scale, on_progress, gpu_id, options)

    def _upscale_sequence(
        self,
        in_dir: Path,
        target: Path,
        frame_rate: str,
        scale: int,
        on_progress: ProgressCallback,
        gpu_id: int | None,
        options: RenderOptions | None = None,
    ) -> None:
        frames = sorted(p for p in in_dir.iterdir() if p.suffix == ".png")
        if not frames:
            raise RuntimeError(f"no PNG frames in {in_dir}")
        info = pipeline.probe(frames[0])
        session = self._session(scale, gpu_id)
        reader = pipeline.open_frame_reader(in_dir / "%08d.png")
        writer = pipeline.open_frame_writer(
            target, info.width * scale, info.height * scale, frame_rate,
            options=RenderOptions(filters=options.filters, encode=options.encode) if options else None,
        )
        done = self._run_stream(session, reader, writer, info.width, info.height,
                                scale, len(frames), on_progress)
        on_progress(done, done)
