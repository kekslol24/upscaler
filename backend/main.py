from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import gpu, pipeline
from .engines import INTERPOLATORS, REGISTRY
from .jobs import INTERPOLATION_FACTORS, INTERPOLATION_MODES, manager
from .paths import FRONTEND_DIR
from .presets import PRESETS

app = FastAPI(title="4K Upscaler")

_UPLOAD_ID_CHARS = frozenset("0123456789abcdef")


def _check_upload_id(upload_id: str) -> None:
    if len(upload_id) != 32 or any(c not in _UPLOAD_ID_CHARS for c in upload_id):
        raise HTTPException(400, "invalid upload id")


class StartJobRequest(BaseModel):
    engine: str
    scale: int
    gpu_id: Optional[int] = None
    #: {"clip": {"start", "duration"}, "filters": {...}, "encode": {...}}
    options: dict[str, Any] = {}


@app.get("/api/gpus")
def list_gpus():
    known = gpu.list_gpus()
    cuda_n = None
    try:
        import torch
        cuda_n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        cuda_n = None
    return {
        "detectable": gpu.detection_available(),
        "rocm": gpu.rocm_available(),
        "cuda_visible_devices": gpu.cuda_visible_devices(),
        "torch_cuda_device_count": cuda_n,
        "gpus": [
            {"index": g.index, "name": g.name, "vendor": g.vendor, "type": g.device_type}
            for g in known
        ],
    }


@app.get("/api/capabilities")
def capabilities():
    """Feature flags the frontend can use without guessing from a 404."""
    return {"pause_resume": True}


@app.get("/api/engines")
def list_engines():
    return [
        {
            "name": engine.name,
            "label": engine.label,
            "description": engine.description,
            "available": engine.is_available(),
            "reason": engine.unavailable_reason(),
            "supported_scales": list(engine.supported_scales),
        }
        for engine in REGISTRY.values()
    ] + ([] if pipeline.ffmpeg_available() else [
        {
            "name": "_ffmpeg",
            "label": "ffmpeg/ffprobe",
            "available": False,
            "reason": "ffmpeg/ffprobe not found on PATH — required for every engine",
            "supported_scales": [],
        }
    ])


@app.get("/api/interpolators")
def list_interpolators():
    return {
        "factors": list(INTERPOLATION_FACTORS),
        "modes": list(INTERPOLATION_MODES),
        "interpolators": [
            {
                "name": interp.name,
                "label": interp.label,
                "description": interp.description,
                "available": interp.is_available(),
                "reason": interp.unavailable_reason(),
            }
            for interp in INTERPOLATORS.values()
        ],
    }


@app.get("/api/encoders")
def list_encoders():
    return [
        {"name": name, "label": pipeline.ENCODER_LABELS.get(name, name),
         "codec": pipeline.encoder_codec(name)}
        for name in pipeline.available_encoders()
    ]


@app.get("/api/presets")
def list_presets():
    return [p.to_public_dict() for p in PRESETS]


@app.get("/api/jobs")
def list_jobs():
    return [job.to_public_dict() for job in manager.list()]


@app.post("/api/upload")
async def upload(file: UploadFile):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    try:
        job = manager.create_job(file.filename or "input.mp4", data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return job.to_public_dict()


@app.post("/api/upload/chunk")
async def upload_chunk(request: Request, upload_id: str, index: int):
    """One piece of a chunked upload, streamed straight to disk. Chunking
    keeps each request under a reverse proxy's body-size limit (RunPod's
    HTTP proxy returns 413 for large single POSTs)."""
    _check_upload_id(upload_id)
    if index < 0 or index > 1_000_000:
        raise HTTPException(400, "invalid chunk index")
    data = await request.body()
    if not data:
        raise HTTPException(400, "empty chunk")
    manager.write_upload_chunk(upload_id, index, data)
    return {"received": index}


@app.post("/api/upload/complete")
def upload_complete(upload_id: str, filename: str):
    _check_upload_id(upload_id)
    try:
        job = manager.create_job_from_upload(upload_id, Path(filename).name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return job.to_public_dict()


@app.post("/api/jobs/{job_id}/start")
def start_job(job_id: str, body: StartJobRequest):
    try:
        job = manager.start(job_id, body.engine, body.scale, body.gpu_id, body.options)
    except KeyError:
        raise HTTPException(404, "job not found")
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc))
    return job.to_public_dict()


@app.post("/api/jobs/{job_id}/pause")
def pause_job(job_id: str):
    try:
        job = manager.pause(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return job.to_public_dict()


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    try:
        job = manager.resume(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return job.to_public_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        job = manager.cancel(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return job.to_public_dict()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.to_public_dict()


@app.get("/api/jobs/{job_id}/input")
def job_input(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return FileResponse(job.input_path)


@app.get("/api/jobs/{job_id}/before")
def job_before(job_id: str):
    """The source side of the compare view for a preview render."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.before_path is None or not job.before_path.exists():
        raise HTTPException(409, "no comparison available for this job")
    return FileResponse(job.before_path)


@app.get("/api/jobs/{job_id}/output")
def job_output(job_id: str, download: bool = False, mp4: bool = False, mkv: bool = False):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.output_path is None or not job.output_path.exists():
        raise HTTPException(409, "output not ready")
    if download:
        # No flags: the user's chosen container. mp4=1 / mkv=1 force a
        # specific one (the secondary download link).
        if mp4:
            path, filename = job.output_path, job.output_filename
        elif mkv and job.subtitle_path is not None and job.subtitle_path.exists():
            path, filename = job.subtitle_path, job.subtitle_filename
        else:
            path, filename = job.download_path, job.download_filename
        return FileResponse(path, filename=filename)
    return FileResponse(job.output_path)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
