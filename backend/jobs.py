"""Job store + single background worker thread.

One job runs at a time by design — this targets a single slow laptop
GPU, not throughput. Job state lives in memory only; artifacts live on
disk under DATA_DIR so a finished output survives a server restart even
though its status entry doesn't.
"""

from __future__ import annotations

import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import gpu, pipeline
from .engines import get_engine

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "jobs"

VALID_STATUSES = ("uploaded", "queued", "running", "done", "error")


@dataclass
class Job:
    id: str
    dir: Path
    input_path: Path
    status: str = "uploaded"
    progress: int = 0
    message: str = ""
    engine: Optional[str] = None
    scale: Optional[int] = None
    gpu_id: Optional[int] = None
    output_path: Optional[Path] = None
    created_at: float = field(default_factory=time.time)

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "engine": self.engine,
            "scale": self.scale,
            "gpu_id": self.gpu_id,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def create_job(self, original_filename: str, data: bytes) -> Job:
        job_id = uuid.uuid4().hex
        job_dir = DATA_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(original_filename).suffix or ".mp4"
        input_path = job_dir / f"input{suffix}"
        input_path.write_bytes(data)

        job = Job(id=job_id, dir=job_dir, input_path=input_path)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, job_id: str, engine_name: str, scale: int, gpu_id: Optional[int] = None) -> Job:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        engine = get_engine(engine_name)  # raises ValueError if unknown
        if not engine.is_available():
            raise RuntimeError(f"engine {engine_name!r} is not available: {engine.unavailable_reason()}")
        if scale not in engine.supported_scales:
            raise ValueError(f"engine {engine_name!r} does not support scale {scale}")
        if job.status not in ("uploaded", "error"):
            raise RuntimeError(f"job {job_id} already {job.status}")
        if gpu_id is not None:
            known_gpus = gpu.list_gpus()
            if known_gpus and gpu_id not in {g.index for g in known_gpus}:
                raise ValueError(f"no Vulkan GPU with index {gpu_id}")

        job.engine = engine_name
        job.scale = scale
        job.gpu_id = gpu_id
        job.status = "queued"
        job.progress = 0
        job.message = "queued"
        self._queue.put(job_id)
        return job

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            try:
                self._process(job)
            except Exception as exc:  # noqa: BLE001 - report any failure on the job
                job.status = "error"
                job.message = str(exc)

    def _process(self, job: Job) -> None:
        job.status = "running"

        job.message = "probing video"
        info = pipeline.probe(job.input_path)

        frames_in = job.dir / "frames_in"
        frames_out = job.dir / "frames_out"

        job.message = "extracting frames"
        pipeline.extract_frames(job.input_path, frames_in)

        engine = get_engine(job.engine)

        def on_progress(done: int, total: int) -> None:
            job.progress = int(done / total * 100) if total else 0
            job.message = f"upscaling frame {done}/{total}"

        engine.upscale_frames(frames_in, frames_out, job.scale, on_progress, gpu_id=job.gpu_id)

        job.message = "encoding output video"
        output_path = job.dir / "output.mp4"
        pipeline.reassemble(
            frames_out,
            output_path,
            fps=info.fps,
            audio_source=job.input_path if info.has_audio else None,
        )
        job.output_path = output_path

        shutil.rmtree(frames_in, ignore_errors=True)
        shutil.rmtree(frames_out, ignore_errors=True)

        job.progress = 100
        job.message = "done"
        job.status = "done"


manager = JobManager()
