# AGENTS.md

## Project

Local video upscaler: FastAPI backend plus a no-build vanilla-JS frontend, driving
ffmpeg and Real-ESRGAN/RIFE GPU engines. See `README.md` for full setup and
deployment details. See `PROGRESS.md` for the current hand-off state and the
4K interpolation performance analysis (encoder-bound CPU fallback).

## Commands

- Run everything from the repo root; imports are flat `backend.*` (no installed
  package, `pyproject.toml`, or task runner).
- Linux first-time setup: `./scripts/setup.sh` (uses sudo; installs ffmpeg/Vulkan,
  creates `.venv`, installs deps, downloads RIFE v4.6 PyTorch weights, and
  downloads ncnn engines only when Vulkan sees a real GPU; CUDA images reuse the
  system PyTorch by enabling venv system site packages). Windows:
  `scripts\setup.ps1`.
- Dev server: `.venv/bin/python -m uvicorn backend.main:app --reload`
  (Windows: `.venv\Scripts\python.exe -m uvicorn backend.main:app --reload`).
  `python launcher.py` picks a free port and opens the browser.
- Dev deps and tests:
  `.venv/bin/python -m pip install -r requirements-dev.txt`
  `.venv/bin/python -m pytest -q`
  Single test: `.venv/bin/python -m pytest tests/test_jobs.py::test_api_end_to_end -q`
- No lint, typecheck, formatter, CI, or pre-commit config exists; pytest is the
  only verification.
- Headless/container: `./scripts/run.sh` binds `0.0.0.0:8000` and keeps data in
  `/workspace/upscaler-data` when `/workspace` exists.
- Per-stage throughput: `.venv/bin/python scripts/bench.py [video]`.

## RIFE implementation (current)

- `rife-v4.6` is the PyTorch/CUDA streaming interpolator
  (`backend/engines/rife_torch.py`, model in `rife_torch_model.py`, weights in
  `models/rife-v4.6/flownet.pkl`). It is the first registry entry and the
  default for presets. `rife-v4.6-vulkan` is the ncnn directory-mode fallback.
- CUDA interpolation never writes PNGs; interpolation-only and
  interpolation + streaming-upscale use streaming paths in `jobs.py`.
- Measured on the RTX 4090 pod: pure RIFE forward ~20 fps at 4K (full-res
  flow), ~45 fps in UHD half-res mode. A 36s 4K 30→60 fps interpolation-only
  render finished in ~2.5 min (~7.5 source fps / ~15 output fps) because
  libx264 encoding was the bottleneck; NVENC was unavailable because the
  container exposed only `compute,utility` driver capabilities.

## Architecture

- `backend/main.py` is the HTTP API; it mounts `frontend/` at `/` as a catch-all
  StaticFiles route, so new API routes must be declared before that mount.
- `backend/jobs.py` owns a single background worker (one render at a time by
  design) and persists jobs in SQLite. `manager` is created at import time.
- `backend/pipeline.py` builds all ffmpeg/ffprobe commands and render options.
  Keep frame rates as exact rational strings (e.g. `24000/1001`) and use
  `scale_frame_rate()` when interpolating.
- Three execution paths: streaming ONNX (`engine.streaming`) pipes
  decoder -> model -> encoder; streaming RIFE (`FrameInterpolator.streaming`)
  pipes decoded RGB frames through PyTorch and back into ffmpeg; interpolation
  plus a streaming upscaler runs those two streaming passes with a temporary
  intermediate video. Renders involving any directory-mode stage (ncnn engines
  or the Vulkan interpolator) still use the chunked pipeline in `jobs.py`
  (5-second chunks, one-frame overlap, bounded disk use). Do not replace it with
  whole-video frame extraction.
- Engines implement `backend/engines/base.py` and are registered in
  `backend/engines/__init__.py`; registry order sets the UI default. Presets in
  `backend/presets.py` only bundle engine preferences, scale, filters and encode
  settings.
- Use `backend/proc.py` (`run`/`Popen`) instead of `subprocess` directly so child
  processes stay hidden in the packaged Windows app.

## Gotchas

- The ONNX engine refuses the CPU provider unless `UPSCALER_ALLOW_CPU=1`; tests
  set this themselves. GPU tests/setup require a CUDA/DirectML provider matching
  the installed runtime.
- `engines_bin/` is downloaded by setup and gitignored; `models/*.onnx` are
  committed. RIFE v4.6 CUDA weights live in `models/rife-v4.6/flownet.pkl` and
  are downloaded by Linux setup. Tests skip when ffmpeg, the ONNX models, the
  RIFE PyTorch/CUDA stack, or the RIFE binary are missing.
- The CUDA interpolator needs `torch` importable from `.venv`. On CUDA images
  setup enables `include-system-site-packages` instead of installing a second
  torch; without torch it reports unavailable and the Vulkan fallback is used.
- `requirements-dev.txt` pulls CPU `onnxruntime` on Linux, which can shadow an
  existing `onnxruntime-gpu` (RunPod/CUDA). In GPU venvs install only `pytest` and
  `httpx` for the tests.
- Headless NVIDIA containers need an X display for the Vulkan ICD: `setup.sh`
  probes `vulkaninfo` through `xvfb-run`, and `run.sh` starts Xvfb when `DISPLAY`
  is unset. The Vulkan engines/RIFE rely on this; the CUDA RIFE path does not.
- Interpolation progress counts source frames, not generated frames. A 2x job
  showing `7.5 fps` is producing ~15 output fps; do not compare that number
  directly with RIFE-only benchmark throughput.
- `NVENC` needs the container's `NVIDIA_DRIVER_CAPABILITIES` to include `video`
  (`all` or `compute,utility,video`). Without it ffmpeg falls back to
  libx264/libx265, which dominates 4K60 render time. The encoder cache key
  includes the capability value, so restarting the pod with new caps re-probes.
- Encoder detection test-encodes each encoder once and caches it at
  `<data dir>/encoders.json`; delete that file to force a re-probe.
- Job data defaults to `./data` in a checkout (`UPSCALER_DATA_DIR` overrides);
  importing/running the app creates `data/jobs/jobs.sqlite3`.
- Desktop packaging is Windows-only: `scripts/build_exe.ps1` plus `upscaler.spec`
  produce a one-folder `dist\Upscaler\` with engines/models/ffmpeg copied beside
  the exe; Inno Setup is optional.
