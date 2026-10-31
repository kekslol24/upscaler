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
  `scripts\setup.ps1`. AMD Linux/WSL2: `./scripts/setup_amd.sh` (or
  `./setup_amd.sh`) installs Mesa Dozen on WSL2, runs `setup.sh`, optionally
  installs the gfx1200 ROCm PyTorch wheel (`--rocm`), then starts the server;
  `--no-start` only sets up.
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

- `rife-v4.6` is the PyTorch streaming interpolator (CUDA on NVIDIA, ROCm on
  AMD: PyTorch exposes the `torch.cuda` API on ROCm). Code in
  `backend/engines/rife_torch.py`, model in `rife_torch_model.py`, weights in
  `models/rife-v4.6/flownet.pkl`. It is the first registry entry and the
  default for presets. `rife-v4.6-vulkan` is the ncnn directory-mode fallback.
- CUDA/ROCm interpolation never writes PNGs; interpolation-only and
  interpolation + streaming-upscale use streaming paths in `jobs.py`.
- Streaming performance work (2026-10-31), all in `rife_torch.py`:
  - GPU→CPU frames go through a reusable **pinned** host buffer
    (`_pinned_host`), not a blocking pageable `.cpu()`.
  - RGB→YUV420p (BT.601 limited) is converted **on the GPU**
    (`_to_yuv_bytes`) and the writer takes `pix_fmt="yuv420p"`; ffmpeg's own
    RGB→YUV conversion cost ~30 ms per 4K frame. Odd dimensions fall back to
    RGB; PNG/directory mode still uses RGB.
  - `_AsyncFrameWriter` writes to ffmpeg stdin from a background thread
    (bounded queue, copies to `bytes` because the pinned buffer is reused),
    overlapping encode with the next forward pass.
- Measured on the RTX 4090 pod, 10 s 4K30→60 libx264 veryfast CRF 20:
  5.4 → 11.0 source fps (10.8 → 22.0 output fps), wall 55 → 27 s. Forward is
  ~23 ms at 4K UHD half-res; the remaining bottleneck is the producer
  (decode/pipe ~19 ms, H2D ~10 ms), not the encoder.

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
  pipes decoded RGB frames through PyTorch and writes YUV420p (or RGB for odd
  dimensions) back into ffmpeg through a pinned buffer and background writer;
  interpolation plus a streaming upscaler runs those two streaming passes with
  a temporary intermediate video. Renders involving any directory-mode stage
  (ncnn engines or the Vulkan interpolator) still use the chunked pipeline in
  `jobs.py` (5-second chunks, one-frame overlap, bounded disk use). Do not
  replace it with whole-video frame extraction.
- Engines implement `backend/engines/base.py` and are registered in
  `backend/engines/__init__.py`; registry order sets the UI default. Presets in
  `backend/presets.py` only bundle engine preferences, scale, filters and encode
  settings.
- Subtitle download: `pipeline.mux_subtitles()` stream-copies the rendered
  MP4's video/audio plus the source's subtitle/font streams into
  `<job>/output_subs.mkv`. `job.subtitle_path` makes that MKV the default
  download (`/api/jobs/{id}/output?download=1`), while `/output` (the player)
  and `?download=1&mp4=1` serve the browser-playable MP4. Previews pass the
  clip window to the mux. No video/audio re-encode.
- Use `backend/proc.py` (`run`/`Popen`) instead of `subprocess` directly so child
  processes stay hidden in the packaged Windows app.

## Gotchas

- The ONNX engine refuses the CPU provider unless `UPSCALER_ALLOW_CPU=1`; tests
  set this themselves. GPU tests/setup require a CUDA/DirectML provider matching
  the installed runtime.
- `engines_bin/` is downloaded by setup and gitignored; `models/*.onnx` are
  committed. RIFE v4.6 weights live in `models/rife-v4.6/flownet.pkl` and are
  downloaded by Linux setup. Tests skip when ffmpeg, the ONNX models, the RIFE
  PyTorch GPU stack (CUDA/ROCm), or the RIFE binary are missing.
- The PyTorch interpolator needs `torch` importable from `.venv`. On CUDA images
  setup enables `include-system-site-packages` instead of installing a second
  torch; without torch it reports unavailable and the Vulkan fallback is used.
  The same code runs on ROCm PyTorch because it implements the `torch.cuda`
  API; install a `torch[device-gfx1200]` wheel (RX 9060 XT) from AMD's index.
  `torch.backends.cudnn.benchmark` is set unconditionally; current ROCm maps it
  to MIOpen autotuning.
- `requirements-dev.txt` pulls CPU `onnxruntime` on Linux, which can shadow an
  existing `onnxruntime-gpu` (RunPod/CUDA). In GPU venvs install only `pytest` and
  `httpx` for the tests. On AMD Linux, ONNX Runtime 1.23+ removed
  `ROCMExecutionProvider`; the current AMD path is the MIGraphX plugin
  (`onnxruntime-ep-migraphx`), which `_GPU_PROVIDERS` does not register yet, so
  the ONNX upscaler is unavailable there (use the Vulkan upscaler). On Windows
  AMD, `onnxruntime-directml` works normally.
- Headless NVIDIA containers need an X display for the Vulkan ICD: `setup.sh`
  probes `vulkaninfo` through `xvfb-run`, and `run.sh` starts Xvfb when `DISPLAY`
  is unset. The Vulkan engines/RIFE rely on this; the CUDA/ROCm RIFE path does
  not.
- Interpolation progress counts source frames, not generated frames. A 2x job
  showing `7.5 fps` is producing ~15 output fps; do not compare that number
  directly with RIFE-only benchmark throughput.
- `NVENC` needs the container's `NVIDIA_DRIVER_CAPABILITIES` to include `video`
  (`all` or `compute,utility,video`). Without it ffmpeg falls back to
  libx264/libx265. Since the GPU YUV + background-writer changes the CPU encoder
  is no longer the dominant 4K bottleneck, but NVENC still helps. The encoder
  cache key includes the capability value, so restarting the pod re-probes.
- AMD RX 9060 XT (gfx1200, RDNA 4): ROCm 10.1 supports it on Linux and Windows
  11. Use a current ROCm/PyTorch build — older ones reported only ~8 GB of the
  16 GB VRAM (ROCm issue #6295) and could segfault past it. It has roughly 1/8
  the FP32 compute and 1/3 the bandwidth of an RTX 4090, so 4K RIFE is likely
  below real-time; 1080p is comfortable. Linux AMD has no VAAPI encoder in the
  candidate list (libx264); Windows AMD can use AMF (`h264_amf`/`hevc_amf`).
- Encoder detection test-encodes each encoder once and caches it at
  `<data dir>/encoders.json`; delete that file to force a re-probe.
- Job data defaults to `./data` in a checkout (`UPSCALER_DATA_DIR` overrides);
  importing/running the app creates `data/jobs/jobs.sqlite3`.
- Desktop packaging is Windows-only: `scripts/build_exe.ps1` plus `upscaler.spec`
  produce a one-folder `dist\Upscaler\` with engines/models/ffmpeg copied beside
  the exe; Inno Setup is optional.
