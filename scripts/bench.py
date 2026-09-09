"""Time every stage of the pipeline on this machine, in isolation, so a
slow render can be pinned on one of them.

    .venv\\Scripts\\python.exe scripts\\bench.py            # synthetic 1080p clip
    .venv\\Scripts\\python.exe scripts\\bench.py video.mkv  # first seconds of a real file

Reports frames/second for: decoding, the ONNX upscaler on its GPU provider
(whole frame and banded, x2), each working video encoder at 4K, PNG
write/read of 4K frames (what the chunked path does), and the RIFE binary
at 1080p and at 4K with a few thread settings.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend import pipeline  # noqa: E402
from backend.engines import INTERPOLATORS, REGISTRY  # noqa: E402
from backend.engines import realesrgan_onnx as onnx_mod  # noqa: E402
from backend.engines import rife_ncnn  # noqa: E402

W, H = 1920, 1080
SECONDS = 4
rows: list[tuple[str, str]] = []


def report(name: str, fps: float | None, note: str = "") -> None:
    value = f"{fps:6.1f} fps" if fps is not None else "   n/a   "
    rows.append((name, f"{value}  {note}".rstrip()))
    print(f"  {name:<44} {value}  {note}".rstrip(), flush=True)


def timed(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def make_source(tmp: Path, given: Path | None) -> Path:
    src = tmp / "src.mp4"
    if given is not None:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-t", str(SECONDS), "-i", str(given),
                        "-map", "0:v:0", "-an", "-vf", f"scale={W}:{H}", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "16", "-pix_fmt", "yuv420p", str(src)], check=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", f"testsrc2=size={W}x{H}:rate=24:duration={SECONDS}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", str(src)],
                       check=True)
    return src


def bench_decode(src: Path) -> list[bytes]:
    frames: list[bytes] = []
    reader = pipeline.open_frame_reader(src)
    t0 = time.perf_counter()
    n = W * H * 3
    while True:
        buf = reader.stdout.read(n)
        if len(buf) < n:
            break
        frames.append(buf)
    reader.wait()
    report("decode 1080p H.264 -> raw RGB", len(frames) / (time.perf_counter() - t0))
    return frames


def bench_model(frames: list[bytes]) -> np.ndarray | None:
    engine = REGISTRY["realesrgan-animevideov3-onnx"]
    if not engine.is_available():
        report("ONNX upscaler x2", None, engine.unavailable_reason() or "")
        return None
    provider = engine._gpu_provider()
    session = engine._session(2, None)
    name = session.get_inputs()[0].name
    arr = [np.frombuffer(f, dtype=np.uint8).reshape(1, H, W, 3) for f in frames]
    n = len(arr) if provider != "CPUExecutionProvider" else 2
    whole = session.run(None, {name: arr[0]})[0]  # warm-up / kernel compile
    t = timed(lambda: [session.run(None, {name: a}) for a in arr[:n]])
    black = int((whole[0].max(axis=(1, 2)) == 0).sum())
    report(f"ONNX upscaler x2, whole frame ({provider})", n / t,
           "output OK" if black == 0 else f"OUTPUT CORRUPT: {black} black rows")
    reference = onnx_mod._infer_frame(session, name, arr[0], 2)
    original = onnx_mod._MAX_OUTPUT_BYTES
    for cap_m in (original / 1024 / 1024, 12, 16, 24):
        cap = int(cap_m * 1024 * 1024)
        onnx_mod._MAX_OUTPUT_BYTES = cap
        try:
            rows_ = onnx_mod._max_band_rows(W, 2)
            out = onnx_mod._infer_frame(session, name, arr[0], 2)
            t = timed(lambda: [onnx_mod._infer_frame(session, name, a, 2) for a in arr[:n]])
        finally:
            onnx_mod._MAX_OUTPUT_BYTES = original
        diff = int(np.abs(out.astype(np.int16) - reference.astype(np.int16)).max())
        tag = "default" if cap == original else f"UPSCALER_BAND_OUTPUT_BYTES={cap}"
        report(f"ONNX upscaler x2, bands of {min(rows_, H)} rows", n / t,
               f"{tag}; {'output OK' if diff <= 1 else f'OUTPUT DIFFERS (max diff {diff})'}")

    # x4 has 4x the output per input row; confirms the limit is on output size
    if provider != "CPUExecutionProvider" and engine._model_path(4).is_file():
        s4 = engine._session(4, None)
        n4 = s4.get_inputs()[0].name
        whole4 = s4.run(None, {n4: arr[0]})[0]
        black4 = int((whole4[0].max(axis=(1, 2)) == 0).sum())
        report("ONNX upscaler x4, whole frame", None,
               "output OK" if black4 == 0 else f"OUTPUT CORRUPT: {black4} of {whole4.shape[1]} rows black")
        band4 = onnx_mod._infer_frame(s4, n4, arr[0], 4)
        black4b = int((band4[0].max(axis=(1, 2)) == 0).sum())
        t = timed(lambda: [onnx_mod._infer_frame(s4, n4, a, 4) for a in arr[: max(2, n // 3)]])
        report(f"ONNX upscaler x4, bands of {onnx_mod._max_band_rows(W, 4)} rows", max(2, n // 3) / t,
               "output OK" if black4b == 0 else f"OUTPUT CORRUPT: {black4b} black rows")
    return reference


def bench_encoders(tmp: Path, frame4k: np.ndarray | None, count: int) -> None:
    if frame4k is None:
        frame4k = np.random.randint(0, 255, (1, H * 2, W * 2, 3), dtype=np.uint8)
    buf = frame4k.tobytes()
    for enc in pipeline.available_encoders():
        out = tmp / f"enc_{enc}.mp4"
        settings = pipeline.EncodeSettings(codec=pipeline.encoder_codec(enc), encoder=enc)
        writer = pipeline.open_frame_writer(out, W * 2, H * 2, "24", options=pipeline.RenderOptions(encode=settings))
        t0 = time.perf_counter()
        try:
            for _ in range(count):
                writer.stdin.write(buf)
            writer.stdin.close()
            writer.wait()
        except BrokenPipeError:
            report(f"encode 4K with {enc}", None, "failed")
            continue
        report(f"encode 4K with {enc}", count / (time.perf_counter() - t0))


def bench_png(tmp: Path, frame4k: np.ndarray | None, count: int) -> Path:
    if frame4k is None:
        frame4k = np.random.randint(0, 255, (1, H * 2, W * 2, 3), dtype=np.uint8)
    buf = frame4k.tobytes()
    png_dir = tmp / "png4k"
    png_dir.mkdir()
    writer = pipeline.open_frame_writer(png_dir / "%08d.png", W * 2, H * 2, "1")
    t0 = time.perf_counter()
    for _ in range(count):
        writer.stdin.write(buf)
    writer.stdin.close()
    writer.wait()
    report("write 4K PNG frames (chunked path)", count / (time.perf_counter() - t0),
           f"{sum(p.stat().st_size for p in png_dir.iterdir()) / count / 1e6:.1f} MB per frame")
    reader = pipeline.open_frame_reader(png_dir / "%08d.png")
    t0 = time.perf_counter()
    n = W * 2 * H * 2 * 3
    got = 0
    while len(reader.stdout.read(n)) == n:
        got += 1
    reader.wait()
    report("read 4K PNG frames", got / (time.perf_counter() - t0))
    return png_dir


def bench_rife(tmp: Path, src: Path, png4k: Path) -> None:
    interp = INTERPOLATORS["rife-v4.6"]
    if not interp.is_available():
        report("RIFE", None, interp.unavailable_reason() or "")
        return
    frames1080 = tmp / "png1080"
    pipeline.extract_frames(src, frames1080, pipeline.RenderOptions(clip=pipeline.Clip(0, 1.0)))
    for label, in_dir, extra in (
        ("RIFE 2x at 1080p, default threads", frames1080, []),
        ("RIFE 2x at 1080p, -j 2:2:4", frames1080, ["-j", "2:2:4"]),
        ("RIFE 2x at 1080p, -j 4:4:4", frames1080, ["-j", "4:4:4"]),
        ("RIFE 2x at 4K with UHD mode (-u)", png4k, []),
    ):
        count = sum(1 for _ in in_dir.iterdir())
        out = tmp / f"rife_{abs(hash(label))}"
        cmd = [str(rife_ncnn.BINARY_PATH), "-i", str(in_dir), "-o", str(out), "-n", str(count * 2 - 1),
               "-m", str(interp.model_dir), "-f", "%08d.png", *extra]
        if in_dir is png4k:
            cmd.append("-u")
        out.mkdir()
        t = timed(lambda: subprocess.run(cmd, cwd=rife_ncnn.ENGINE_DIR, capture_output=True, check=True))
        report(label, (count * 2 - 1) / t, f"{count} frames in, output frames/s")


def main() -> None:
    given = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    tmp = Path(tempfile.mkdtemp(prefix="upscaler-bench-"))
    try:
        print(f"benchmark, scratch dir {tmp}")
        print(f"ffmpeg: {pipeline._ffmpeg_version()}")
        pipeline.available_encoders.cache_clear()
        print(f"working encoders: {pipeline.available_encoders()}")
        for name, err in pipeline.ENCODER_ERRORS.items():
            print(f"  {name} failed: {err}")
        try:
            import onnxruntime as ort
            print(f"onnxruntime {ort.__version__}, providers {ort.get_available_providers()}")
        except ImportError:
            print("onnxruntime not installed")
        print()
        src = make_source(tmp, given)
        frames = bench_decode(src)
        frame4k = bench_model(frames)
        count = 48 if frame4k is not None else 24
        bench_encoders(tmp, frame4k, count)
        png4k = bench_png(tmp, frame4k, 24)
        bench_rife(tmp, src, png4k)
        print("\nsummary")
        for name, value in rows:
            print(f"  {name:<44} {value}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    os.environ.setdefault(onnx_mod._ALLOW_CPU_ENV, "1")
    main()
