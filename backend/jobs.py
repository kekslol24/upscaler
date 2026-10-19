"""Job store + single background worker thread.

One job runs at a time by design — this targets a single slow laptop
GPU, not throughput. Jobs are persisted as JSON rows in a SQLite file
next to the artifacts, so finished outputs are still listed after a
server restart (a job that was mid-render is marked as interrupted).

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

VALID_STATUSES = ("uploaded", "queued", "running", "done", "error", "cancelled")
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
    created_at: float = field(default_factory=time.time)
    cancel_requested: threading.Event = field(default_factory=threading.Event, repr=False)

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
        return RenderOptions(
            clip=self.clip,
            filters=Filters(
                deinterlace=bool(f.get("deinterlace", False)),
                denoise=int(f.get("denoise", 0)),
                sharpen=bool(f.get("sharpen", False)),
            ),
            encode=EncodeSettings(
                codec=str(e.get("codec", "h264")),
                encoder=str(e.get("encoder", "auto")),
                quality=int(e.get("quality", 18)),
            ),
        )

    @property
    def output_filename(self) -> str:
        stem = Path(self.original_filename).stem or "video"
        parts = []
        if self.scale and self.scale > 1:
            parts.append(f"x{self.scale}")
        interp = self.interpolation
        if interp:
            parts.append(f"{interp.mode}{interp.factor}x")
        if self.clip:
            parts.append("preview")
        return "_".join([stem, *parts]) + ".mp4"

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

    def __call__(self, done: float, verb: str) -> None:
        job = self.job
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


class JobManager:
    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.data_dir / "jobs.sqlite3"
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._init_db()
        self._load()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # -- persistence ----------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=10)

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
            if job.status in ("queued", "running"):
                job.status = "error"
                job.message = "interrupted by server restart"
                job.progress = 0
                shutil.rmtree(job.dir / "work", ignore_errors=True)
                self._save(job)
            self._jobs[job.id] = job

    # -- API ------------------------------------------------------------

    def create_job(self, original_filename: str, data: bytes) -> Job:
        job_id = uuid.uuid4().hex
        job_dir = self.data_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(original_filename).suffix or ".mp4"
        input_path = job_dir / f"input{suffix}"
        input_path.write_bytes(data)

        try:
            info = pipeline.probe(input_path)
        except Exception as exc:  # noqa: BLE001 - ffprobe failure or no video stream
            shutil.rmtree(job_dir, ignore_errors=True)
            raise ValueError(f"not a readable video: {exc}") from exc

        job = Job(
            id=job_id, dir=job_dir, input_path=input_path,
            original_filename=original_filename,
            duration=info.duration, width=info.width, height=info.height,
        )
        with self._lock:
            self._jobs[job_id] = job
        self._save(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

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
        if gpu_id is not None:
            known_gpus = gpu.list_gpus()
            if known_gpus and gpu_id not in {g.index for g in known_gpus}:
                raise ValueError(f"no GPU with index {gpu_id}")

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
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid render options: {exc}") from exc

        job.engine = engine_name
        job.scale = scale
        job.gpu_id = gpu_id
        job.status = "queued"
        job.progress = 0
        job.message = "queued"
        job.output_path = None
        job.cancel_requested.clear()
        self._save(job)
        self._queue.put(job_id)
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status not in ("queued", "running"):
            raise RuntimeError(f"job {job_id} is {job.status}, nothing to cancel")
        job.cancel_requested.set()
        if job.status == "queued":
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

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None or job.status != "queued":
                continue
            try:
                self._process(job)
            except JobCancelled:
                self._discard_partial(job)
                self._finish(job, "cancelled", "cancelled")
            except Exception as exc:  # noqa: BLE001 - report any failure on the job
                self._discard_partial(job)
                self._finish(job, "error", str(exc))

    def _discard_partial(self, job: Job) -> None:
        if job.output_path is not None:
            job.output_path.unlink(missing_ok=True)
            job.output_path = None
        shutil.rmtree(job.dir / "work", ignore_errors=True)

    def _process(self, job: Job) -> None:
        job.status = "running"
        job.message = "probing video"
        self._save(job)

        info = pipeline.probe(job.input_path)
        options = job.render_options()
        interp = job.interpolation
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
        else:
            self._run_chunked(job, info, engine, interp, options, output_path, progress)

        if job.cancel_requested.is_set():
            raise JobCancelled()

        if options.clip is not None:
            # previews are short: build the matching source clip before reporting done
            job.before_path = self._before_video(job, options.clip)
            job.progress = 100
            self._finish(job, "done", "done")
            return
        # a full render is done — and downloadable — as soon as the file is;
        # the browser-playable copy of the source for the compare view can
        # take minutes for a long episode, so it is built afterwards
        job.before_path = None
        job.progress = 100
        self._finish(job, "done", "done")
        try:
            job.before_path = self._before_video(job, None)
        except Exception as exc:  # noqa: BLE001 - the render itself succeeded
            job.message = f"done (no comparison copy: {exc})"
        self._save(job)

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

    def _before_video(self, job: Job, clip: Clip | None) -> Path:
        """The source side of the compare view: the matching window as a
        browser-playable proxy, or the upload itself when that already is."""
        if clip is not None:
            proxy = job.dir / "preview_before.mp4"
            job.message = "preparing comparison clip"
            pipeline.make_proxy(job.input_path, proxy, clip)
            return proxy
        if pipeline.is_browser_playable(job.input_path):
            return job.input_path
        proxy = job.dir / "input_proxy.mp4"
        if not proxy.exists():
            job.message = "done · preparing the source copy for the compare view"
            pipeline.make_proxy(job.input_path, proxy)
            job.message = "done"
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
