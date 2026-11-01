"""JobManager persistence, cancellation and the API, end to end on the
CPU ONNX provider with a tiny clip."""

import json
import time
from pathlib import Path

import pytest

from backend import pipeline
from backend.engines import INTERPOLATORS, REGISTRY, realesrgan_onnx as engine_mod

pytestmark = pytest.mark.skipif(not pipeline.ffmpeg_available(), reason="ffmpeg not installed")

ENGINE = "realesrgan-animevideov3-onnx"


@pytest.fixture
def cpu_engine(monkeypatch):
    pytest.importorskip("onnxruntime")
    if not (engine_mod.MODELS_DIR / "realesr-animevideov3-x2.onnx").is_file():
        pytest.skip("ONNX models not exported")
    # These tests are deliberately CPU-only; on a machine with a GPU
    # provider the env var alone would still select the GPU.
    monkeypatch.setattr(
        engine_mod.RealEsrganOnnxEngine,
        "_gpu_provider",
        lambda self: "CPUExecutionProvider",
    )


@pytest.fixture
def manager(tmp_path):
    from backend.jobs import JobManager
    return JobManager(data_dir=tmp_path / "jobs")


def _wait(manager, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        if job.status not in ("queued", "running"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"job still {manager.get(job_id).status} after {timeout}s")


def test_upload_probes_and_rejects_garbage(manager, tiny_video):
    job = manager.create_job("tiny.mkv", tiny_video.read_bytes())
    assert (job.width, job.height, round(job.duration)) == (64, 48, 1)
    with pytest.raises(ValueError):
        manager.create_job("nope.mp4", b"not a video")


def test_pick_gpu_prefers_idle_worker(monkeypatch, tmp_path):
    import backend.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "_worker_gpu_ids", lambda: [0, 1])
    manager = jobs_mod.JobManager(data_dir=tmp_path / "jobs")
    assert manager._pick_gpu() == 0
    manager._running[0] = 1
    assert manager._pick_gpu() == 1
    manager._running[1] = 1
    assert manager._pick_gpu() in (0, 1)


def test_chunked_upload_assembles_chunks(manager, tiny_video):
    data = tiny_video.read_bytes()
    upload_id = "a" * 32
    half = len(data) // 2
    manager.write_upload_chunk(upload_id, 0, data[:half])
    manager.write_upload_chunk(upload_id, 1, data[half:])
    job = manager.create_job_from_upload(upload_id, "tiny.mkv")
    assert job.input_path.name == "input.mkv"
    assert (job.width, job.height) == (64, 48)
    assert not manager.upload_dir(upload_id).exists()


def test_chunked_upload_rejects_missing_chunk(manager, tiny_video):
    upload_id = "b" * 32
    manager.write_upload_chunk(upload_id, 0, tiny_video.read_bytes())
    manager.write_upload_chunk(upload_id, 2, b"x")
    with pytest.raises(ValueError, match="missing"):
        manager.create_job_from_upload(upload_id, "tiny.mkv")


def test_output_format_mkv_creates_mkv_download(manager, tiny_video, cpu_engine):
    job = manager.create_job("tiny.mkv", tiny_video.read_bytes())
    manager.start(job.id, ENGINE, 2, options={
        "encode": {"encoder": "libx264", "format": "mkv"},
    })
    job = _wait(manager, job.id)
    assert job.status == "done", job.message
    assert job.output_format == "mkv"
    assert job.subtitle_path is not None and job.subtitle_path.exists()
    assert job.download_filename == "tiny_x2.mkv"
    assert job.download_path == job.subtitle_path
    assert pipeline.probe(job.subtitle_path).width == 128


def test_preview_then_full_render_and_persistence(manager, tiny_video, cpu_engine, tmp_path):
    job = manager.create_job("tiny.mkv", tiny_video.read_bytes())

    manager.start(job.id, ENGINE, 2, options={
        "clip": {"start": 0.25, "duration": 0.5},
        "filters": {"denoise": 1},
        "encode": {"encoder": "libx264", "quality": 24},
    })
    job = _wait(manager, job.id)
    assert job.status == "done", job.message
    assert job.output_path.name == "preview.mp4"
    assert job.output_filename == "tiny_x2_preview.mp4"
    info = pipeline.probe(job.output_path)
    assert (info.width, info.height) == (128, 96) and info.has_audio
    assert job.before_path.name == "preview_before.mp4"

    manager.start(job.id, ENGINE, 2, options={"encode": {"encoder": "libx264"}})
    job = _wait(manager, job.id)
    assert job.status == "done", job.message
    assert job.output_path.name == "output.mp4"
    assert pipeline.probe(job.output_path).estimated_frames == 12
    # A full render has no before/after comparison and must not spend time
    # building a source proxy (that would delay the next queued job).
    assert job.before_path is None
    assert not job.to_public_dict()["before_ready"]
    assert job.to_public_dict()["output_path"]

    # a fresh manager on the same directory sees the finished job
    from backend.jobs import JobManager
    reloaded = JobManager(data_dir=tmp_path / "jobs")
    again = reloaded.get(job.id)
    assert again is not None and again.status == "done" and again.output_path.exists()
    assert again.to_public_dict()["has_output"]


def test_interrupted_job_is_marked_on_reload(tmp_path, tiny_video):
    from backend.jobs import JobManager
    m = JobManager(data_dir=tmp_path / "jobs")
    job = m.create_job("tiny.mkv", tiny_video.read_bytes())
    job.status = "running"
    m._save(job)
    reloaded = JobManager(data_dir=tmp_path / "jobs")
    assert reloaded.get(job.id).status == "error"
    assert "restart" in reloaded.get(job.id).message


def test_cancel_running_job(manager, tiny_video, cpu_engine, monkeypatch):
    # make every frame slow enough that the cancel lands mid-render
    real_infer = engine_mod._infer_frame

    def slow_infer(*args, **kwargs):
        time.sleep(0.3)
        return real_infer(*args, **kwargs)

    monkeypatch.setattr(engine_mod, "_infer_frame", slow_infer)
    job = manager.create_job("tiny.mkv", tiny_video.read_bytes())
    manager.start(job.id, ENGINE, 2, options={"encode": {"encoder": "libx264"}})
    deadline = time.time() + 30
    while manager.get(job.id).status != "running" and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)
    manager.cancel(job.id)
    job = _wait(manager, job.id)
    assert job.status == "cancelled"
    assert job.output_path is None and not (job.dir / "output.mp4").exists()


def test_start_validation(manager, tiny_video, cpu_engine):
    job = manager.create_job("tiny.mkv", tiny_video.read_bytes())
    with pytest.raises(ValueError):
        manager.start(job.id, ENGINE, 2, options={"clip": {"start": 5, "duration": 1}})
    with pytest.raises(ValueError):
        manager.start(job.id, ENGINE, 2, options={"encode": {"encoder": "h264_nonexistent"}})
    with pytest.raises(ValueError):
        manager.start(job.id, ENGINE, 2, options={"encode": {"format": "avi"}})
    with pytest.raises(ValueError):
        manager.start(job.id, "no-such-engine", 2)


def test_api_end_to_end(tiny_video, cpu_engine, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import backend.jobs as jobs_mod
    from backend.jobs import JobManager

    monkeypatch.setattr(jobs_mod, "manager", JobManager(data_dir=tmp_path / "jobs"))
    import backend.main as main_mod
    monkeypatch.setattr(main_mod, "manager", jobs_mod.manager)
    client = TestClient(main_mod.app)

    assert any(e["name"] == "libx264" for e in client.get("/api/encoders").json())
    assert client.get("/api/capabilities").json()["pause_resume"] is True
    presets = client.get("/api/presets").json()
    assert presets and presets[0]["engines"]

    res = client.post("/api/upload", files={"file": ("tiny.mkv", tiny_video.read_bytes())})
    assert res.status_code == 200, res.text
    job_id = res.json()["id"]

    res = client.post(f"/api/jobs/{job_id}/start", json={
        "engine": ENGINE, "scale": 2,
        "options": {"clip": {"start": 0, "duration": 0.5}, "encode": {"encoder": "libx264"}},
    })
    assert res.status_code == 200, res.text
    job = _wait(jobs_mod.manager, job_id)
    assert job.status == "done", job.message

    listed = client.get("/api/jobs").json()
    assert listed[0]["id"] == job_id and listed[0]["has_output"]
    assert client.get(f"/api/jobs/{job_id}/before").status_code == 200
    res = client.get(f"/api/jobs/{job_id}/output", params={"download": True})
    assert res.status_code == 200
    assert "tiny_x2_preview.mp4" in res.headers["content-disposition"]
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409


def test_pause_and_resume_running_job(manager, tiny_video, cpu_engine, monkeypatch):
    real_infer = engine_mod._infer_frame

    def slow_infer(*args, **kwargs):
        time.sleep(0.35)
        return real_infer(*args, **kwargs)

    monkeypatch.setattr(engine_mod, "_infer_frame", slow_infer)
    job = manager.create_job("tiny.mkv", tiny_video.read_bytes())
    manager.start(job.id, ENGINE, 2, options={"encode": {"encoder": "libx264"}})

    deadline = time.time() + 30
    while time.time() < deadline:
        job = manager.get(job.id)
        if job.status == "running" and job.progress > 0:
            break
        time.sleep(0.05)
    assert job.status == "running" and job.progress > 0

    manager.pause(job.id)
    deadline = time.time() + 10
    while time.time() < deadline:
        job = manager.get(job.id)
        if job.status == "paused" and job.message.startswith("paused ·"):
            break
        time.sleep(0.05)
    assert job.status == "paused" and job.message.startswith("paused ·")
    paused_progress = job.progress
    time.sleep(1.0)
    job = manager.get(job.id)
    assert job.status == "paused" and job.progress == paused_progress

    manager.resume(job.id)
    job = _wait(manager, job.id, timeout=60)
    assert job.status == "done", job.message
    assert job.progress == 100


def test_api_pause_and_resume(tiny_video, cpu_engine, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import backend.jobs as jobs_mod
    from backend.jobs import JobManager

    real_infer = engine_mod._infer_frame

    def slow_infer(*args, **kwargs):
        time.sleep(0.35)
        return real_infer(*args, **kwargs)

    monkeypatch.setattr(engine_mod, "_infer_frame", slow_infer)
    monkeypatch.setattr(jobs_mod, "manager", JobManager(data_dir=tmp_path / "jobs"))
    import backend.main as main_mod
    monkeypatch.setattr(main_mod, "manager", jobs_mod.manager)
    client = TestClient(main_mod.app)

    res = client.post("/api/upload", files={"file": ("tiny.mkv", tiny_video.read_bytes())})
    job_id = res.json()["id"]
    res = client.post(f"/api/jobs/{job_id}/start", json={
        "engine": ENGINE, "scale": 2,
        "options": {"encode": {"encoder": "libx264"}},
    })
    assert res.status_code == 200, res.text

    deadline = time.time() + 30
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] == "running" and body["progress"] > 0:
            break
        time.sleep(0.05)
    assert body["status"] == "running"

    res = client.post(f"/api/jobs/{job_id}/pause")
    assert res.status_code == 200, res.text
    assert res.json()["status"] == "paused"
    res = client.post(f"/api/jobs/{job_id}/resume")
    assert res.status_code == 200, res.text
    assert res.json()["status"] == "running"

    job = _wait(jobs_mod.manager, job_id, timeout=60)
    assert job.status == "done", job.message


@pytest.mark.skipif(
    not REGISTRY["anime4k-v4"].is_available(),
    reason="Anime4K/libplacebo not available",
)
def test_anime4k_job_end_to_end(manager, tiny_video):
    job = manager.create_job("tiny.mkv", tiny_video.read_bytes())
    manager.start(job.id, "anime4k-v4", 2, options={
        "encode": {"encoder": "libx264"},
    })
    assert manager.get(job.id).pausable is False
    with pytest.raises(RuntimeError, match="cannot pause"):
        manager.pause(job.id)
    job = _wait(manager, job.id)
    assert job.status == "done", job.message
    info = pipeline.probe(job.output_path)
    assert (info.width, info.height) == (128, 96)
    assert info.frame_rate == "12/1" and info.has_audio


@pytest.mark.skipif(
    not REGISTRY["anime4k-v4"].is_available()
    or not INTERPOLATORS["rife-v4.6"].is_available(),
    reason="Anime4K and/or RIFE PyTorch not available",
)
def test_anime4k_after_rife_job(manager, tiny_video):
    job = manager.create_job("tiny.mkv", tiny_video.read_bytes())
    manager.start(job.id, "anime4k-v4", 2, options={
        "interpolate": {"engine": "rife-v4.6", "factor": 2, "mode": "smooth"},
        "encode": {"encoder": "libx264"},
    })
    job = _wait(manager, job.id)
    assert job.status == "done", job.message
    info = pipeline.probe(job.output_path)
    assert (info.width, info.height) == (128, 96)
    assert info.frame_rate == "24/1" and info.has_audio
