"""Job store + one background worker per GPU.

Each GPU has its own worker thread and queue, so a multi-GPU machine can
render one episode on each GPU concurrently. Jobs started with "Auto" go
to the least-loaded GPU; an explicit gpu_id pins one. A machine with no
usable GPU gets a single worker, preserving the old serial behaviour.
Jobs are persisted as JSON rows in a SQLite file next to the artifacts,
so finished outputs are still listed after a server restart (a job that
was mid-render is marked as interrupted).

A job is one uploaded source; it can be rendered more than once — a
few preview clips to compare settings, then the full render.

Two execution paths:

- streaming: upscale only, with a streaming engine — frames go decoder
  -> model -> encoder over pipes, nothing on disk.
- chunked: anything involving a directory-mode stage (the ncnn engines,
  frame interpolation). The source is decoded once and cut into short
  chunks of PNG frames; each chunk is upscaled, interpolated and encoded
  to a video segment, then its frames are deleted, so disk use stays
  bounded regardless of the source length. Segments are concatenated
  without re-encoding at the end. When interpolating, each chunk starts
  with the previous chunk's last frame so the interpolator sees every
  pair; the duplicate output frame is dropped.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import gpu, pipeline
from .engines import NO_UPSCALE, get_engine, get_interpolator
from .engines.base import JobCancelled
from .paths import JOBS_DIR
from .pipeline import Clip, EncodeSettings, Filters, RenderOptions

DATA_DIR = JOBS_DIR

VALID_STATUSES = ("uploaded", "queued", "running", "paused", "done", "error", "cancelled")
#: a render can be (re)started from any of these
RESTARTABLE = ("uploaded", "done", "error", "cancelled")

#: how often progress ticks are flushed to disk
_PROGRESS_SAVE_INTERVAL = 2.0
#: chunk length for the directory pipeline; 5 s of 1080p is ~250 frames,
#: a few GB of PNGs at 4K — the disk high-water mark per job
CHUNK_SECONDS = 5.0
INTERPOLATION_FACTORS = (2, 3, 4)
INTERPOLATION_MODES = ("smooth", "slowmo")


@dataclass(frozen=True)
class Interpolation:
    engine: str
    factor: int
    #: "smooth" multiplies the frame rate (same duration, audio kept);
    #: "slowmo" keeps the frame rate so the clip plays factor x slower
    mode: str


@dataclass
class Job:
    id: str
    dir: Path
    input_path: Path
    original_filename: str = ""
    status: str = "uploaded"
    progress: int = 0
    message: str = ""
    engine: Optional[str] = None
    scale: Optional[int] = None
    gpu_id: Optional[int] = None
    #: render options as submitted (clip / filters / encode / interpolate), JSON-shaped
    options: dict[str, Any] = field(default_factory=dict)
    #: source metadata from ffprobe at upload time
    duration: float = 0.0
    width: int = 0
    height: int = 0
    output_path: Optional[Path] = None
    #: browser-playable "before" video for the compare view
    before_path: Optional[Path] = None
    #: MKV with the source's subtitles/fonts stream-copied in (download)
    subtitle_path: Optional[Path] = None
    created_at: float = field(default_factory=time.time)
    cancel_requested: threading.Event = field(default_factory=threading.Event, repr=False)
    #: cooperative pause: the render loop blocks on `resume_event` after
    #: finishing the frame that is in flight, so no GPU work is discarded.
    pause_requested: threading.Event = field(default_factory=threading.Event, repr=False)
    resume_event: threading.Event = field(default_factory=threading.Event, repr=False)
    #: True when pause was requested on a job that had already started, so
    #: resume only has to wake the worker instead of re-queueing it.
    paused_while_running: bool = field(default=False, repr=False)
    #: False for engines that cannot cooperate with the pause signal
    pausable: bool = True

    @property
    def clip(self) -> Clip | None:
        clip = self.options.get("clip")
        return Clip(float(clip["start"]), float(clip["duration"])) if clip else None

    @property
    def interpolation(self) -> Interpolation | None:
        i = self.options.get("interpolate")
        if not i or not i.get("engine"):
            return None
        return Interpolation(str(i["engine"]), int(i.get("factor", 2)), str(i.get("mode", "smooth")))

    def render_options(self) -> RenderOptions:
        f = self.options.get("filters") or {}
        e = self.options.get("encode") or {}
        fmt = str(e.get("format", "mp4")).lower()
        if fmt not in ("mp4", "mkv"):
            raise ValueError("output format must be 'mp4' or 'mkv'")
        return RenderOptions(
            clip=self.clip,
            filters=Filters(
                deinterlace=bool(f.get("deinterlace", False)),
                denoise=int(f.get("denoise", 0)),
                sharpen=bool(f.get("sharpen", False)),
            ),
            encode=EncodeSettings(
                codec=str(e.get("codec", "hevc")),
                encoder=str(e.get("encoder", "auto")),
                quality=int(e.get("quality", 18)),
                format=fmt,
            ),
        )

    @property
    def output_format(self) -> str:
        fmt = str((self.options.get("encode") or {}).get("format", "mp4")).lower()
        return fmt if fmt in ("mp4", "mkv") else "mp4"

    @property
    def subtitle_format(self) -> str:
        fmt = str((self.options.get("encode") or {}).get("subtitles", "copy")).lower()
        if fmt not in pipeline.SUBTITLE_FORMATS:
            raise ValueError(
                f"subtitles must be one of {', '.join(pipeline.SUBTITLE_FORMATS)}"
            )
        return fmt

    def _output_name(self, ext: str) -> str:
        stem = Path(self.original_filename).stem or "video"
        parts = []
        if self.scale and self.scale > 1:
            parts.append(f"x{self.scale}")
        interp = self.interpolation
        if interp:
            parts.append(f"{interp.mode}{interp.factor}x")
        if self.clip:
            parts.append("preview")
        return "_".join([stem, *parts]) + ext

    @property
    def output_filename(self) -> str:
        return self._output_name(".mp4")

    @property
    def subtitle_filename(self) -> str:
        return self._output_name(".mkv")

    @property
    def download_path(self) -> Optional[Path]:
        if (self.output_format == "mkv"
                and self.subtitle_path is not None
                and self.subtitle_path.exists()):
            return self.subtitle_path
        return self.output_path

    @property
    def download_filename(self) -> str:
        if (self.output_format == "mkv"
                and self.subtitle_path is not None
                and self.subtitle_path.exists()):
            return self.subtitle_filename
        return self.output_filename

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.original_filename,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "engine": self.engine,
            "scale": self.scale,
            "gpu_id": self.gpu_id,
            "options": self.options,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "has_output": self.output_path is not None and self.output_path.exists(),
            "output_path": str(self.output_path) if self.output_path and self.output_path.exists() else None,
            "before_ready": self.before_path is not None and self.before_path.exists(),
            "output_filename": self.output_filename,
            "has_subtitle_output": self.subtitle_path is not None and self.subtitle_path.exists(),
            "subtitle_output_filename": (
                self.subtitle_filename
                if self.subtitle_path is not None and self.subtitle_path.exists()
                else None
            ),
            "download_filename": self.download_filename,
            "output_format": self.output_format,
            "pausable": self.pausable,
            "created_at": self.created_at,
        }

    # -- persistence ----------------------------------------------------

    def to_record(self) -> dict:
        return {
            "id": self.id,
            "dir": str(self.dir),
            "input_path": str(self.input_path),
            "original_filename": self.original_filename,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "engine": self.engine,
            "scale": self.scale,
            "gpu_id": self.gpu_id,
            "options": self.options,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "output_path": str(self.output_path) if self.output_path else None,
            "before_path": str(self.before_path) if self.before_path else None,
            "subtitle_path": str(self.subtitle_path) if self.subtitle_path else None,
            "pausable": self.pausable,
            "created_at": self.created_at,
        }

    @classmethod
    def from_record(cls, rec: dict) -> "Job":
        return cls(
            id=rec["id"],
            dir=Path(rec["dir"]),
            input_path=Path(rec["input_path"]),
            original_filename=rec.get("original_filename", ""),
            status=rec.get("status", "uploaded"),
            progress=int(rec.get("progress", 0)),
            message=rec.get("message", ""),
            engine=rec.get("engine"),
            scale=rec.get("scale"),
            gpu_id=rec.get("gpu_id"),
            options=rec.get("options") or {},
            duration=float(rec.get("duration", 0.0)),
            width=int(rec.get("width", 0)),
            height=int(rec.get("height", 0)),
            output_path=Path(rec["output_path"]) if rec.get("output_path") else None,
            before_path=Path(rec["before_path"]) if rec.get("before_path") else None,
            subtitle_path=Path(rec["subtitle_path"]) if rec.get("subtitle_path") else None,
            pausable=bool(rec.get("pausable", True)),
            created_at=float(rec.get("created_at", time.time())),
        )


@dataclass
class _Chunk:
    idx: int
    dir: Path
    #: frames read from the source in this chunk
    n_new: int
    #: including the carried-over last frame of the previous chunk
    n_total: int
    has_carry: bool
    last: bool
    #: directory of frames ready for the final stage
    frames: Path | None = None


def _format_eta(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


class _Progress:
    """Turns stage callbacks into job progress + a throughput/ETA message,
    checking for cancellation on every tick."""

    def __init__(self, job: Job, total: int, save: Callable[[Job], None]) -> None:
        self.job = job
        self.total = max(total, 1)
        self.save = save
        self.started = time.monotonic()
        self.last_save = self.started
        self._pause_lock = threading.Lock()

    def _wait_if_paused(self, done: float, verb: str) -> None:
        """Block here while the user has the render paused.

        The worker parks between frames rather than being killed, so the
        encoder stays open and resuming continues with the next frame.
        """
        job = self.job
        with self._pause_lock:
            if not job.pause_requested.is_set():
                return
            if job.status != "paused":
                job.status = "paused"
            job.message = f"paused · {verb} {int(done)}/{self.total}"
            self.save(job)
            while job.pause_requested.is_set() and not job.cancel_requested.is_set():
                job.resume_event.wait(0.5)
            job.resume_event.clear()
            if job.cancel_requested.is_set():
                return
            if job.status == "paused":
                job.status = "running"
                job.message = f"{verb} frame {int(done)}/{self.total}"
                self.save(job)

    def __call__(self, done: float, verb: str) -> None:
        job = self.job
        if job.pause_requested.is_set():
            self._wait_if_paused(done, verb)
        if job.cancel_requested.is_set():
            raise JobCancelled()
        elapsed = time.monotonic() - self.started
        job.progress = max(0, min(99, int(done / self.total * 100)))
        msg = f"{verb} frame {int(done)}/{self.total}"
        if done >= 1 and elapsed > 2:
            rate = done / elapsed
            msg += f" · {rate:.1f} fps · ~{_format_eta((self.total - done) / rate)} left"
        job.message = msg
        if time.monotonic() - self.last_save > _PROGRESS_SAVE_INTERVAL:
            self.last_save = time.monotonic()
            self.save(job)


def _worker_gpu_ids() -> list[int | None]:
    """GPU indices to run one worker each; [None] means a single worker
    with no GPU pin (tests, CPU-only machines)."""
    ids: list[int | None] = []
    try:
        import torch
        if torch.cuda.is_available():
            ids = list(range(torch.cuda.device_count()))
    except Exception:  # noqa: BLE001 - torch is optional
        ids = []
    if not ids:
        try:
            ids = [g.index for g in gpu.list_gpus()]
        except Exception:  # noqa: BLE001 - detection is best-effort
            ids = []
    if not ids:
        ids = [None]
    cap = os.environ.get("UPSCALER_WORKERS", "").strip()
    if cap:
        try:
            limit = int(cap)
        except ValueError:
            limit = 0
        if limit > 0:
            ids = ids[:limit]
    return ids


class JobManager:
    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.data_dir / "jobs.sqlite3"
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._gpu_ids = _worker_gpu_ids()
        self._queues: dict[int | None, "queue.Queue[str]"] = {
            gpu_id: queue.Queue() for gpu_id in self._gpu_ids
        }
        self._running: dict[int | None, int] = {
            gpu_id: 0 for gpu_id in self._gpu_ids
        }
        self._init_db()
        self._load()
        self._cleanup_uploads()
        self._workers = [
            threading.Thread(
                target=self._worker_loop,
                args=(gpu_id, self._queues[gpu_id]),
                daemon=True,
            )
            for gpu_id in self._gpu_ids
        ]
        for worker in self._workers:
            worker.start()

    # -- persistence ----------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS jobs ("
                " id TEXT PRIMARY KEY, created_at REAL NOT NULL, data TEXT NOT NULL)"
            )

    def _save(self, job: Job) -> None:
        rec = json.dumps(job.to_record())
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, created_at, data) VALUES (?, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (job.id, job.created_at, rec),
            )

    def _load(self) -> None:
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM jobs ORDER BY created_at").fetchall()
        for (data,) in rows:
            job = Job.from_record(json.loads(data))
            if not job.input_path.exists():
                continue
            if job.status in ("queued", "running", "paused"):
                job.status = "error"
                job.message = "interrupted by server restart"
                job.progress = 0
                shutil.rmtree(job.dir / "work", ignore_errors=True)
                self._save(job)
            self._jobs[job.id] = job

    # -- API ------------------------------------------------------------

    def _register_job(self, original_filename: str, job_dir: Path, input_path: Path) -> Job:
        original_filename = Path(original_filename).name or "input.mp4"
        try:
            info = pipeline.probe(input_path)
        except Exception as exc:  # noqa: BLE001 - ffprobe failure or no video stream
            shutil.rmtree(job_dir, ignore_errors=True)
            raise ValueError(f"not a readable video: {exc}") from exc

        job = Job(
            id=job_dir.name, dir=job_dir, input_path=input_path,
            original_filename=original_filename,
            duration=info.duration, width=info.width, height=info.height,
        )
        with self._lock:
            self._jobs[job.id] = job
        self._save(job)
        return job

    def create_job(self, original_filename: str, data: bytes) -> Job:
        job_id = uuid.uuid4().hex
        job_dir = self.data_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(original_filename).suffix or ".mp4"
        input_path = job_dir / f"input{suffix}"
        input_path.write_bytes(data)
        return self._register_job(original_filename, job_dir, input_path)

    def upload_dir(self, upload_id: str) -> Path:
        return self.data_dir / "uploads" / upload_id

    def write_upload_chunk(self, upload_id: str, index: int, data: bytes) -> None:
        """Append one chunk of a large upload to disk (no whole-file RAM)."""
        target = self.upload_dir(upload_id)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{index:06d}.part").write_bytes(data)

    def create_job_from_upload(self, upload_id: str, filename: str) -> Job:
        """Assemble the uploaded chunks into a job input and probe it."""
        upload = self.upload_dir(upload_id)
        parts = sorted(upload.glob("*.part"))
        if not parts:
            raise ValueError("no chunks uploaded")
        if [int(part.stem) for part in parts] != list(range(len(parts))):
            raise ValueError("missing upload chunk")

        job_id = uuid.uuid4().hex
        job_dir = self.data_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename).suffix or ".mp4"
        input_path = job_dir / f"input{suffix}"
        with input_path.open("wb") as out:
            for part in parts:
                with part.open("rb") as src:
                    shutil.copyfileobj(src, out, 1024 * 1024)
        shutil.rmtree(upload, ignore_errors=True)
        return self._register_job(filename, job_dir, input_path)

    def _cleanup_uploads(self) -> None:
        root = self.data_dir / "uploads"
        if not root.exists():
            return
        cutoff = time.time() - 24 * 3600
        for item in root.iterdir():
            try:
                if item.is_dir() and item.stat().st_mtime < cutoff:
                    shutil.rmtree(item, ignore_errors=True)
            except OSError:
                pass

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def delete(self, job_id: str) -> None:
        """Remove a finished job, its database row and its files.

        Active jobs are refused: cancel them first so a worker is not
        writing into a directory that is being deleted."""
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status not in RESTARTABLE:
            raise RuntimeError(f"job {job_id} is {job.status}; cancel it first")
        with self._lock:
            self._jobs.pop(job_id, None)
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        finally:
            try:
                job.dir.resolve().relative_to(self.data_dir.resolve())
            except ValueError:
                pass
            else:
                shutil.rmtree(job.dir, ignore_errors=True)

    def clear(self) -> int:
        """Delete every finished/uploaded job. Returns the count removed."""
        with self._lock:
            ids = [job.id for job in self._jobs.values() if job.status in RESTARTABLE]
        deleted = 0
        for job_id in ids:
            try:
                self.delete(job_id)
            except (KeyError, RuntimeError):
                continue
            deleted += 1
        return deleted

    def _pick_gpu(self) -> int | None:
        """Least busy worker: queued plus currently running jobs."""
        return min(
            self._queues,
            key=lambda g: self._queues[g].qsize() + self._running[g],
        )

    def start(
        self,
        job_id: str,
        engine_name: str,
        scale: int,
        gpu_id: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Job:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        options = dict(options or {})
        interp_opts = options.get("interpolate") or {}
        if interp_opts and not interp_opts.get("engine"):
            interp_opts = {}
            options.pop("interpolate", None)

        engine = None
        interpolator = None
        if engine_name == NO_UPSCALE:
            if not interp_opts:
                raise ValueError("nothing to do: no upscaling engine and no frame interpolation")
            scale = 1
        else:
            engine = get_engine(engine_name)  # raises ValueError if unknown
            if not engine.is_available():
                raise RuntimeError(f"engine {engine_name!r} is not available: {engine.unavailable_reason()}")
            if scale not in engine.supported_scales:
                raise ValueError(f"engine {engine_name!r} does not support scale {scale}")
        if interp_opts:
            interpolator = get_interpolator(str(interp_opts["engine"]))
            if not interpolator.is_available():
                raise RuntimeError(
                    f"interpolator {interpolator.name!r} is not available: {interpolator.unavailable_reason()}")
            factor, mode = int(interp_opts.get("factor", 2)), str(interp_opts.get("mode", "smooth"))
            if factor not in INTERPOLATION_FACTORS or mode not in INTERPOLATION_MODES:
                raise ValueError("interpolation factor must be 2-4 and mode 'smooth' or 'slowmo'")
            options["interpolate"] = {"engine": interpolator.name, "factor": factor, "mode": mode}
        if job.status not in RESTARTABLE:
            raise RuntimeError(f"job {job_id} is {job.status}")
        if gpu_id is None:
            gpu_id = self._pick_gpu()
        elif gpu_id not in self._queues:
            known_gpus = {g.index for g in gpu.list_gpus()}
            if known_gpus and gpu_id not in known_gpus:
                raise ValueError(f"no GPU with index {gpu_id}")
            raise ValueError(
                f"GPU {gpu_id} is not served by this process "
                f"(workers: {list(self._queues)})"
            )

        clip = options.get("clip")
        if clip:
            start, duration = float(clip.get("start", 0)), float(clip.get("duration", 5))
            if duration <= 0 or start < 0 or (job.duration and start >= job.duration):
                raise ValueError("preview window is outside the video")
            options["clip"] = {"start": start, "duration": duration}
        encode = options.get("encode") or {}
        if encode.get("encoder", "auto") != "auto" and encode["encoder"] not in pipeline.available_encoders():
            raise ValueError(f"encoder {encode['encoder']!r} is not available on this machine")
        try:
            job.options = options
            job.render_options()
            job.subtitle_format
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid render options: {exc}") from exc

        job.engine = engine_name
        job.scale = scale
        job.gpu_id = gpu_id
        job.pausable = not (
            (engine is not None and getattr(engine, "pausable", True) is False)
            or (interp_opts and interpolator is not None
                and getattr(interpolator, "pausable", True) is False)
        )
        job.status = "queued"
        job.progress = 0
        job.message = "queued"
        job.output_path = None
        # Drop the previous run's derived files so a stale subtitle copy or
        # comparison proxy is never served for the new settings.
        if job.subtitle_path is not None:
            job.subtitle_path.unlink(missing_ok=True)
            job.subtitle_path = None
        if job.before_path is not None:
            job.before_path.unlink(missing_ok=True)
            job.before_path = None
        job.cancel_requested.clear()
        job.pause_requested.clear()
        job.resume_event.clear()
        job.paused_while_running = False
        self._save(job)
        self._queues[gpu_id].put(job_id)
        return job

    def pause(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if not job.pausable:
            raise RuntimeError(
                f"job {job_id} uses an engine that cannot pause; cancel it instead"
            )
        if job.status == "queued":
            # Drop out of the queue; resume() puts it back. The worker may
            # still pick up the stale id and will skip it because the status
            # is no longer "queued".
            job.paused_while_running = False
            job.pause_requested.set()
            job.status = "paused"
            job.message = "paused"
            self._save(job)
            return job
        if job.status != "running":
            raise RuntimeError(f"job {job_id} is {job.status}, nothing to pause")
        job.paused_while_running = True
        job.pause_requested.set()
        job.resume_event.clear()
        job.status = "paused"
        job.message = "pausing…"
        self._save(job)
        return job

    def resume(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status != "paused":
            raise RuntimeError(f"job {job_id} is {job.status}, not paused")
        job.pause_requested.clear()
        job.resume_event.set()
        if job.paused_while_running:
            job.paused_while_running = False
            job.status = "running"
            job.message = "resuming"
            self._save(job)
            return job
        job.status = "queued"
        job.message = "queued"
        gpu_id = job.gpu_id if job.gpu_id in self._queues else self._pick_gpu()
        job.gpu_id = gpu_id
        self._save(job)
        self._queues[gpu_id].put(job.id)
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status not in ("queued", "running", "paused"):
            raise RuntimeError(f"job {job_id} is {job.status}, nothing to cancel")
        job.cancel_requested.set()
        job.pause_requested.clear()
        job.resume_event.set()  # wake a paused render so it can see the cancel
        if job.status in ("queued", "paused") and not job.paused_while_running:
            self._finish(job, "cancelled", "cancelled")
        else:
            job.message = "cancelling…"
        return job

    # -- worker ---------------------------------------------------------

    def _finish(self, job: Job, status: str, message: str) -> None:
        job.status = status
        job.message = message
        if status != "done":
            job.progress = 0
        self._save(job)

    def _worker_loop(self, gpu_id: int | None, work: "queue.Queue[str]") -> None:
        while True:
            job_id = work.get()
            job = self.get(job_id)
            if job is None or job.status != "queued":
                continue
            with self._lock:
                self._running[gpu_id] += 1
            try:
                self._process(job)
            except JobCancelled:
                self._discard_partial(job)
                self._finish(job, "cancelled", "cancelled")
            except Exception as exc:  # noqa: BLE001 - report any failure on the job
                self._discard_partial(job)
                self._finish(job, "error", str(exc))
            finally:
                with self._lock:
                    self._running[gpu_id] -= 1

    def _discard_partial(self, job: Job) -> None:
        if job.output_path is not None:
            job.output_path.unlink(missing_ok=True)
            job.output_path = None
        if job.subtitle_path is not None:
            job.subtitle_path.unlink(missing_ok=True)
            job.subtitle_path = None
        shutil.rmtree(job.dir / "work", ignore_errors=True)

    def _mux_subtitles(self, job: Job, output_path: Path) -> str | None:
        """Stream-copy the source's subtitles/fonts into an MKV beside the
        browser-playable MP4. Returns an error string on failure; the MP4
        stays valid and downloadable either way."""
        need_mkv = job.output_format == "mkv" or pipeline.has_subtitles(job.input_path)
        if not need_mkv:
            return None
        target = job.dir / ("preview_subs.mkv" if job.clip else "output_subs.mkv")
        job.message = "muxing subtitles"
        try:
            pipeline.mux_subtitles(
                output_path, job.input_path, target, job.clip, job.subtitle_format,
            )
        except Exception as exc:  # noqa: BLE001 - the render itself succeeded
            return str(exc)
        job.subtitle_path = target
        return None

    def _process(self, job: Job) -> None:
        if job.cancel_requested.is_set():
            raise JobCancelled()
        if job.pause_requested.is_set():
            # The pause landed after the worker dequeued the job but before
            # it started; park it the same way a running pause does.
            job.paused_while_running = True
            job.status = "paused"
            job.message = "paused"
            self._save(job)
            while job.pause_requested.is_set() and not job.cancel_requested.is_set():
                job.resume_event.wait(0.5)
            job.resume_event.clear()
            if job.cancel_requested.is_set():
                raise JobCancelled()
        job.paused_while_running = False
        job.status = "running"
        job.message = "probing video"
        self._save(job)

        info = pipeline.probe(job.input_path)
        options = job.render_options()
        interp = job.interpolation
        interpolator = get_interpolator(interp.engine) if interp else None
        engine = None if job.engine == NO_UPSCALE else get_engine(job.engine)
        output_path = job.dir / ("preview.mp4" if options.clip else "output.mp4")
        job.output_path = output_path
        progress = _Progress(job, info.frames_in(options.clip), self._save)

        if interp is None and engine is not None and engine.streaming:
            job.message = "upscaling"
            engine.upscale_video(
                job.input_path, output_path, info, job.scale,
                lambda d, t: progress(d, "upscaling + encoding"),
                gpu_id=job.gpu_id, options=options,
            )
        elif (interpolator is not None and interpolator.streaming
              and (engine is None or engine.streaming)):
            self._run_streaming(
                job, info, engine, interpolator, interp, options, output_path, progress,
            )
        else:
            self._run_chunked(job, info, engine, interp, options, output_path, progress)

        if job.cancel_requested.is_set():
            raise JobCancelled()

        subtitle_error = self._mux_subtitles(job, output_path)
        if job.cancel_requested.is_set():
            raise JobCancelled()
        done_message = (
            "done" if not subtitle_error
            else f"done (subtitle copy failed: {subtitle_error})"
        )

        if options.clip is not None:
            # previews are short: build the matching source clip before reporting done
            job.before_path = self._before_video(job, options.clip)
        else:
            # A full render has no before/after view: building a source proxy
            # would hold the GPU worker (and the next queued job) hostage for
            # minutes just to enable a comparison nobody needs.
            job.before_path = None
        job.progress = 100
        self._finish(job, "done", done_message)

    def _run_streaming(
        self,
        job: Job,
        info: pipeline.VideoInfo,
        engine,
        interpolator,
        interp: Interpolation,
        options: RenderOptions,
        output_path: Path,
        progress: _Progress,
    ) -> None:
        if engine is None:
            job.message = "interpolating"
            interpolator.interpolate_video(
                job.input_path, output_path, info, interp.factor, interp.mode,
                lambda d, t: progress(d, "interpolating"),
                gpu_id=job.gpu_id, options=options,
            )
            return

        work = job.dir / "work"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        intermediate = work / "interpolated.mp4"
        total = max(info.frames_in(options.clip), 1)
        interp_options = RenderOptions(
            clip=options.clip,
            filters=Filters(
                deinterlace=options.filters.deinterlace,
                denoise=options.filters.denoise,
                sharpen=False,
            ),
            encode=EncodeSettings(codec="h264", encoder="auto", quality=18, preset="veryfast"),
        )
        job.message = "interpolating"
        interpolator.interpolate_video(
            job.input_path, intermediate, info, interp.factor, interp.mode,
            lambda d, t: progress(d * 0.5, "interpolating"),
            gpu_id=job.gpu_id, options=interp_options,
        )

        temp_info = pipeline.probe(intermediate)
        temp_total = max(temp_info.frames_in(None), 1)
        up_options = RenderOptions(
            clip=None,
            filters=Filters(
                deinterlace=False,
                denoise=0,
                sharpen=options.filters.sharpen,
            ),
            encode=options.encode,
        )

        def on_upscale(done: int, _total: int) -> None:
            fraction = min(done / temp_total, 1.0)
            progress(total * 0.5 + total * 0.5 * fraction, "upscaling + encoding")

        job.message = "upscaling + encoding"
        engine.upscale_video(
            intermediate, output_path, temp_info, job.scale, on_upscale,
            gpu_id=job.gpu_id, options=up_options,
        )
        shutil.rmtree(work, ignore_errors=True)

    def _run_chunked(
        self,
        job: Job,
        info: pipeline.VideoInfo,
        engine,
        interp: Interpolation | None,
        options: RenderOptions,
        output_path: Path,
        progress: _Progress,
    ) -> None:
        scale = job.scale if engine is not None else 1
        interpolator = get_interpolator(interp.engine) if interp else None
        out_rate = info.frame_rate
        if interp and interp.mode == "smooth":
            out_rate = pipeline.scale_frame_rate(info.frame_rate, interp.factor)

        work = job.dir / "work"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        frames_per_chunk = max(1, round(CHUNK_SECONDS * info.fps))

        def run_upscale(chunk: _Chunk, src: Path, dst: Path, cb) -> None:
            engine.upscale_frames(src, dst, scale, cb, gpu_id=job.gpu_id)

        def run_interp(chunk: _Chunk, src: Path, dst: Path, cb) -> None:
            target = (chunk.n_total - 1) * interp.factor + 1
            interpolator.interpolate_frames(src, dst, target, cb, gpu_id=job.gpu_id)
            if chunk.has_carry:
                # first output == carried frame, already emitted by the previous chunk
                _drop_first_frame(dst)
            _renumber(dst)

        stages = _stage_order(engine, interpolator, run_upscale, run_interp)
        # With two stages, the producer thread runs the first one (on the
        # next chunk) while this thread runs the second (on the current one):
        # interpolation is mostly PNG I/O and the upscaler is GPU compute, so
        # they overlap well. Progress/ETA are reported from the final stage.
        ahead = stages[:-1]
        final_verb, final_run = stages[-1]
        # a streaming upscaler as the last stage encodes the segment itself
        # instead of writing 4K PNGs that get read straight back
        direct_encode = final_verb == "upscaling" and getattr(engine, "streaming", False)

        ready: "queue.Queue[_Chunk | None]" = queue.Queue(maxsize=1)
        abort = threading.Event()
        producer_error: list[BaseException] = []

        def check_cancel(done: int, total: int) -> None:
            if abort.is_set() or job.cancel_requested.is_set():
                raise JobCancelled()

        def produce() -> None:
            reader = pipeline.open_frame_reader(job.input_path, options)
            frame_bytes = info.width * info.height * 3
            carry: bytes | None = None
            idx = 0
            try:
                while not abort.is_set():
                    frames: list[bytes] = []
                    while len(frames) < frames_per_chunk:
                        buf = reader.stdout.read(frame_bytes)
                        if len(buf) < frame_bytes:
                            break
                        frames.append(buf)
                    if not frames:
                        break
                    chunk_frames = ([carry] if carry is not None else []) + frames
                    chunk = _Chunk(
                        idx=idx, dir=work / f"chunk{idx:04d}", n_new=len(frames),
                        n_total=len(chunk_frames), has_carry=carry is not None,
                        last=len(frames) < frames_per_chunk,
                    )
                    frames_dir = chunk.dir / "in"
                    self._write_pngs(frames_dir, chunk_frames, info)
                    for verb, run in ahead:
                        # only the very first chunk reports here; afterwards the
                        # final stage owns the progress line
                        cb = (lambda d, t, v=verb: progress(chunk.n_new * 0.5 * d / max(t, 1), v)) if idx == 0 else check_cancel
                        nxt = chunk.dir / verb
                        run(chunk, frames_dir, nxt, cb)
                        shutil.rmtree(frames_dir, ignore_errors=True)
                        frames_dir = nxt
                    chunk.frames = frames_dir
                    if abort.is_set():
                        break
                    ready.put(chunk)
                    carry = frames[-1] if interpolator is not None else None
                    idx += 1
                    if chunk.last:
                        break
                reader.stdout.close()
                err = reader.stderr.read().decode(errors="replace").strip()
                if reader.wait() != 0:
                    raise RuntimeError(f"ffmpeg decode failed ({reader.returncode}): {err}")
            except BaseException as exc:  # noqa: BLE001 - handed to the consumer
                producer_error.append(exc)
                reader.kill()
                reader.wait()
            finally:
                ready.put(None)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        segments: list[Path] = []
        source_done = 0
        try:
            while True:
                chunk = ready.get()
                if chunk is None:
                    break
                segment = work / f"seg{chunk.idx:04d}.mp4"
                n_new = chunk.n_new

                def on_final(done: int, total: int) -> None:
                    progress(source_done + n_new * 0.95 * done / max(total, 1), final_verb)

                if direct_encode:
                    engine.upscale_frames_to_video(chunk.frames, segment, out_rate, scale, on_final,
                                                   gpu_id=job.gpu_id, options=options)
                else:
                    out_dir = chunk.dir / final_verb
                    final_run(chunk, chunk.frames, out_dir, on_final)
                    progress(source_done + n_new * 0.98, "encoding")
                    pipeline.reassemble(out_dir, segment, out_rate, None, options)
                segments.append(segment)
                shutil.rmtree(chunk.dir, ignore_errors=True)
                source_done += n_new
                progress(source_done, final_verb)
        except BaseException:
            abort.set()
            # keep the hand-off queue drained until the producer has exited,
            # or it blocks on its next put and never sees the abort
            while producer.is_alive():
                try:
                    ready.get(timeout=0.1)
                except queue.Empty:
                    pass
            raise
        producer.join()
        if producer_error:
            raise producer_error[0]

        if not segments:
            raise RuntimeError("no frames decoded from the source")
        job.message = "joining segments"
        keep_audio = info.has_audio and not (interp and interp.mode == "slowmo")
        pipeline.concat_segments(segments, output_path, job.input_path if keep_audio else None, options)
        shutil.rmtree(work, ignore_errors=True)

    @staticmethod
    def _write_pngs(target_dir: Path, frames: list[bytes], info: pipeline.VideoInfo) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        writer = pipeline.open_frame_writer(target_dir / "%08d.png", info.width, info.height, "1")
        try:
            for buf in frames:
                writer.stdin.write(buf)
            writer.stdin.close()
        except BrokenPipeError:
            pass
        err = writer.stderr.read().decode(errors="replace").strip()
        if writer.wait() != 0:
            raise RuntimeError(f"writing chunk frames failed ({writer.returncode}): {err}")

    def _before_video(self, job: Job, clip: Clip) -> Path:
        """The source side of the compare view: the preview window as a
        browser-playable proxy. Full renders skip this entirely."""
        proxy = job.dir / "preview_before.mp4"
        job.message = "preparing comparison clip"
        pipeline.make_proxy(job.input_path, proxy, clip)
        return proxy


def _stage_order(engine, interpolator, run_upscale, run_interp) -> list[tuple[str, Callable]]:
    """Which comes first, upscaling or interpolation.

    Interpolation cost scales with pixel count, upscaling with frame count.
    With a fast (streaming) upscaler it is much cheaper to interpolate at
    source resolution and upscale the extra frames than to run RIFE on 4K
    frames; with the slow Vulkan upscalers doubling its frame count costs
    more than interpolating at output resolution does.
    """
    up = ("upscaling", run_upscale) if engine is not None else None
    ip = ("interpolating", run_interp) if interpolator is not None else None
    if up and ip:
        return [ip, up] if engine.streaming else [up, ip]
    return [s for s in (up, ip) if s]


def _sorted_pngs(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix == ".png")


def _drop_first_frame(directory: Path) -> None:
    _sorted_pngs(directory)[0].unlink()


def _renumber(directory: Path) -> None:
    """Rename frames to a gapless %08d sequence from 0 (what reassemble()
    reads), via temporary names so nothing is overwritten mid-way."""
    files = _sorted_pngs(directory)
    temps = []
    for i, path in enumerate(files):
        tmp = directory / f"tmp{i:08d}.png.part"
        path.rename(tmp)
        temps.append(tmp)
    for i, tmp in enumerate(temps):
        tmp.rename(directory / f"{i:08d}.png")


manager = JobManager()
