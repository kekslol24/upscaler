from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import gpu, pipeline
from .engines import REGISTRY
from .jobs import manager

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="4K Upscaler")


class StartJobRequest(BaseModel):
    engine: str
    scale: int
    gpu_id: Optional[int] = None


@app.get("/api/gpus")
def list_gpus():
    return {
        "detectable": gpu.vulkaninfo_available(),
        "gpus": [
            {"index": g.index, "name": g.name, "vendor": g.vendor, "type": g.device_type}
            for g in gpu.list_gpus()
        ],
    }


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


@app.post("/api/upload")
async def upload(file: UploadFile):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    job = manager.create_job(file.filename or "input.mp4", data)
    return {"job_id": job.id}


@app.post("/api/jobs/{job_id}/start")
def start_job(job_id: str, body: StartJobRequest):
    try:
        job = manager.start(job_id, body.engine, body.scale, body.gpu_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc))
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


@app.get("/api/jobs/{job_id}/output")
def job_output(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.output_path is None or not job.output_path.exists():
        raise HTTPException(409, "output not ready")
    return FileResponse(job.output_path)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
