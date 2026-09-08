# 4K Upscaler

Local video upscaler with a pluggable model "harness". Upload a short clip,
pick an engine, get a before/after slider.

## Setup (Arch Linux or WSL2 Ubuntu/Debian)

```bash
./scripts/setup.sh
```

This installs `ffmpeg`/Vulkan via pacman or apt, creates a Python venv,
installs Python deps, and downloads the Real-ESRGAN ncnn-vulkan engine binary
(runs on Intel/AMD/NVIDIA GPUs via Vulkan — no CUDA needed).

On WSL2, Vulkan reaches your Windows GPU through Mesa's Dozen (D3D12)
driver, which the script installs. If `vulkaninfo` only lists `llvmpipe`
(CPU fallback) afterwards, update your Windows GPU driver and run
`wsl --update` from Windows. Keeping the project on the Linux filesystem
(e.g. `~/upscaler`) instead of `/mnt/c` is noticeably faster.

## Run

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000, upload a short clip, pick an engine and scale,
and wait — 4K upscaling on integrated graphics is slow, that's expected.

## Adding a new model engine

Implement `backend/engines/base.py::UpscaleEngine` in a new file under
`backend/engines/`, then register an instance in `backend/engines/__init__.py`.
The rest of the pipeline (frame extraction, job queue, API, frontend) is
engine-agnostic — any engine that turns a directory of frame PNGs into a
directory of upscaled frame PNGs works.
