"""The chunked stage pipeline: chunk overlap, interpolation frame counts,
smooth vs slow-motion output, and interpolate-only renders — driven by a
fake interpolator so it runs without the RIFE binary. The real RIFE
engine gets a test that runs only where the binary is installed."""

import shutil
import subprocess
import time

import pytest

from backend import pipeline
from backend.engines import INTERPOLATORS, NO_UPSCALE
from backend.engines import realesrgan_onnx as engine_mod
from backend.engines import rife_ncnn
from backend.engines.base import FrameInterpolator
import backend.jobs as jobs_mod

pytestmark = pytest.mark.skipif(not pipeline.ffmpeg_available(), reason="ffmpeg not installed")

ENGINE = "realesrgan-animevideov3-onnx"


class BlendInterpolator(FrameInterpolator):
    """Stand-in for RIFE with the same contract: N frames in, target_count
    evenly spaced frames out, endpoints equal to the inputs. In-between
    frames are copies of the nearest earlier input frame."""

    name = "fake-blend"
    label = "fake"

    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def is_available(self):
        return True

    def interpolate_frames(self, in_dir, out_dir, target_count, on_progress, gpu_id=None):
        frames = sorted(p for p in in_dir.iterdir() if p.suffix == ".png")
        self.calls.append((len(frames), target_count))
        out_dir.mkdir(parents=True, exist_ok=True)
        span = len(frames) - 1
        for i in range(target_count):
            src = frames[min(span, (i * span) // max(target_count - 1, 1))]
            shutil.copy(src, out_dir / f"{i:08d}.png")
            on_progress(i + 1, target_count)


@pytest.fixture
def fake_interpolator(monkeypatch):
    interp = BlendInterpolator()
    monkeypatch.setitem(INTERPOLATORS, interp.name, interp)
    return interp


@pytest.fixture
def cpu_engine(monkeypatch):
    pytest.importorskip("onnxruntime")
    if not (engine_mod.MODELS_DIR / "realesr-animevideov3-x2.onnx").is_file():
        pytest.skip("ONNX models not exported")
    monkeypatch.setenv(engine_mod._ALLOW_CPU_ENV, "1")


@pytest.fixture
def manager(tmp_path):
    return jobs_mod.JobManager(data_dir=tmp_path / "jobs")


def _wait(manager, job_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        if job.status not in ("queued", "running"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"job still {manager.get(job_id).status}")


def _video_packets(path):
    return int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip())


def test_scale_frame_rate():
    assert pipeline.scale_frame_rate("24000/1001", 2) == "48000/1001"
    assert pipeline.scale_frame_rate("25/1", 4) == "100/1"


def test_smooth_interpolation_doubles_frames_and_keeps_audio(manager, tiny_video, fake_interpolator, monkeypatch):
    # 12 source frames at 12 fps; 0.4 s chunks -> 5-frame chunks, so the
    # overlap logic runs across three chunks
    monkeypatch.setattr(jobs_mod, "CHUNK_SECONDS", 0.4)
    job = manager.create_job("tiny.mkv", tiny_video.read_bytes())
    manager.start(job.id, NO_UPSCALE, 1, options={
        "interpolate": {"engine": "fake-blend", "factor": 2, "mode": "smooth"},
        "encode": {"encoder": "libx264"},
    })
    job = _wait(manager, job.id)
    assert job.status == "done", job.message
    # chunks of 5 / 5(+1 carry) / 2(+1 carry): each asks for (n-1)*2+1 frames
    assert fake_interpolator.calls == [(5, 9), (6, 11), (3, 5)]
    info = pipeline.probe(job.output_path)
    assert info.frame_rate == "24/1" and info.has_audio
    assert _video_packets(job.output_path) == (12 - 1) * 2 + 1
    assert (info.width, info.height) == (64, 48)
    assert job.output_filename == "tiny_smooth2x.mp4"
    assert not (job.dir / "work").exists()


def test_slowmo_keeps_fps_and_drops_audio(manager, tiny_video, fake_interpolator):
    job = manager.create_job("tiny.mkv", tiny_video.read_bytes())
    manager.start(job.id, NO_UPSCALE, 1, options={
        "interpolate": {"engine": "fake-blend", "factor": 4, "mode": "slowmo"},
        "clip": {"start": 0, "duration": 0.5},
        "encode": {"encoder": "libx264"},
    })
    job = _wait(manager, job.id)
    assert job.status == "done", job.message
    info = pipeline.probe(job.output_path)
    assert info.frame_rate == "12/1" and not info.has_audio
    assert _video_packets(job.output_path) == (6 - 1) * 4 + 1
    assert abs(info.duration - 21 / 12) < 0.1


def test_upscale_then_interpolate_via_chunked_path(manager, tiny_video, fake_interpolator, cpu_engine, monkeypatch):
    monkeypatch.setattr(jobs_mod, "CHUNK_SECONDS", 0.5)
    job = manager.create_job("tiny.mkv", tiny_video.read_bytes())
    manager.start(job.id, ENGINE, 2, options={
        "interpolate": {"engine": "fake-blend", "factor": 2, "mode": "smooth"},
        "filters": {"sharpen": True},
        "encode": {"encoder": "libx264"},
    })
    job = _wait(manager, job.id)
    assert job.status == "done", job.message
    info = pipeline.probe(job.output_path)
    assert (info.width, info.height) == (128, 96)
    assert _video_packets(job.output_path) == 23
    assert job.output_filename == "tiny_x2_smooth2x.mp4"


def test_start_rejects_nothing_to_do_and_bad_factor(manager, tiny_video, fake_interpolator):
    job = manager.create_job("tiny.mkv", tiny_video.read_bytes())
    with pytest.raises(ValueError):
        manager.start(job.id, NO_UPSCALE, 1)
    with pytest.raises(ValueError):
        manager.start(job.id, NO_UPSCALE, 1, options={"interpolate": {"engine": "fake-blend", "factor": 7}})
    with pytest.raises(ValueError):
        manager.start(job.id, NO_UPSCALE, 1, options={"interpolate": {"engine": "no-such", "factor": 2}})


def test_cancel_during_chunked_render(manager, tiny_video, fake_interpolator, monkeypatch):
    real = fake_interpolator.interpolate_frames

    def slow(*args, **kwargs):
        time.sleep(1.0)
        return real(*args, **kwargs)

    monkeypatch.setattr(fake_interpolator, "interpolate_frames", slow)
    monkeypatch.setattr(jobs_mod, "CHUNK_SECONDS", 0.25)
    job = manager.create_job("tiny.mkv", tiny_video.read_bytes())
    manager.start(job.id, NO_UPSCALE, 1, options={
        "interpolate": {"engine": "fake-blend", "factor": 2, "mode": "smooth"},
        "encode": {"encoder": "libx264"},
    })
    while manager.get(job.id).status != "running":
        time.sleep(0.05)
    time.sleep(0.3)
    manager.cancel(job.id)
    job = _wait(manager, job.id)
    assert job.status == "cancelled"
    assert not (job.dir / "work").exists() and not (job.dir / "output.mp4").exists()


@pytest.mark.skipif(not INTERPOLATORS["rife-v4.6"].is_available(), reason="rife-ncnn-vulkan not installed")
def test_real_rife_directory_mode_contract(tmp_path, tiny_video):
    """Documents the assumption the chunk pipeline makes about the binary:
    `-n` frames come out, sorted names are playback order."""
    frames = tmp_path / "in"
    pipeline.extract_frames(tiny_video, frames, pipeline.RenderOptions(clip=pipeline.Clip(0, 0.25)))
    out = tmp_path / "out"
    progress = []
    rife_ncnn.RifeNcnnInterpolator("rife-v4.6", "rife", "rife-v4.6").interpolate_frames(
        frames, out, 5, lambda d, t: progress.append((d, t)))
    assert len(sorted(out.glob("*.png"))) == 5
    assert progress[-1] == (5, 5)
