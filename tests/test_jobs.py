"""JobManager persistence, cancellation and the API, end to end on the
CPU ONNX provider with a tiny clip."""

import json
import time
from pathlib import Path

import pytest

from backend import pipeline
from backend.engines import realesrgan_onnx as engine_mod

pytestmark = pytest.mark.skipif(not pipeline.ffmpeg_available(), reason="ffmpeg not installed")

ENGINE = "realesrgan-animevideov3-onnx"


@pytest.fixture
def cpu_engine(monkeypatch):
    pytest.importorskip("onnxruntime")
    if not (engine_mod.MODELS_DIR / "realesr-animevideov3-x2.onnx").is_file():
        pytest.skip("ONNX models not exported")
    monkeypatch.setenv(engine_mod._ALLOW_CPU_ENV, "1")


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
    # MKV upload isn't browser playable, so a proxy is produced for the compare view
    assert job.before_path.name == "input_proxy.mp4"

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
