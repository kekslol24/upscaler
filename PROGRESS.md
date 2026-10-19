# RIFE CUDA progress — 2026-10-19

## Current state

- Server is **stopped**. The last `./scripts/run.sh` process exited; an orphan
  `Xvfb :99` (PID 30964 at the time of writing) may remain.
- Job data lives in `/workspace/upscaler-data`.
- Test suite: `31 passed` (`.venv/bin/python -m pytest -q`).
- No frontend redesign has been started.
- All changes are still **uncommitted** at the time this file was written
  (see `git status`).
- The first 36s 4K interpolation-only job (`5a80093a...`) reached 100% and
  produced a valid 2167-frame 4K60 output, but a later cancel request marked
  the job `cancelled` and `_discard_partial()` deleted the output file. The
  source upload is still present.

## What was implemented

- `backend/engines/rife_torch.py`: new default `rife-v4.6` PyTorch/CUDA
  interpolator. It reads/writes raw RGB frames through ffmpeg pipes instead of
  PNG directories.
- `backend/engines/rife_torch_model.py`: vendored minimal RIFE v4.6 inference
  network (MIT, from hzwer/Practical-RIFE).
- `models/rife-v4.6/flownet.pkl`: RIFE v4.6 CUDA weights, downloaded from the
  Hugging Face `Bash2X/RIFE-Models` mirror; `LICENSE` is included beside it.
- `backend/engines/base.py`: `FrameInterpolator.streaming` and
  `interpolate_video()` optional streaming contract.
- `backend/engines/__init__.py`: `rife-v4.6` now means CUDA/PyTorch;
  `rife-v4.6-vulkan` is the old ncnn fallback. CUDA is first/default.
- `backend/jobs.py`: streaming branches for interpolation-only and
  interpolation + streaming upscale. The combined path writes a temporary
  interpolated H.264 video, then runs the normal streaming ONNX upscaler.
  Directory-mode chunking remains for ncnn/Vulkan combinations.
- `backend/pipeline.py`: optional encoder preset; libx264 now defaults to
  `veryfast`; encoder cache key includes `NVIDIA_DRIVER_CAPABILITIES`.
- `scripts/setup.sh`: reuses the image's system PyTorch via venv
  `include-system-site-packages`, and downloads RIFE v4.6 PyTorch weights.
- `scripts/run.sh`, `backend/gpu.py`, `README.md`, `AGENTS.md`: retained the
  earlier headless Vulkan/Xvfb fixes from this session.
- `frontend/app.js`: preset interpolation falls back to an available
  interpolator instead of selecting an unavailable engine.
- `tests/test_stages.py`: real CUDA RIFE streaming test and CUDA RIFE +
  streaming ONNX test; the ncnn contract test now targets
  `rife-v4.6-vulkan`.

## Why the UI showed ~7 fps

- The interpolation progress line counts **source frames**, not generated
  frames. A 2x smooth render showing `7.5 fps` is producing about
  **15 output fps**.
- The direct RIFE benchmark (20–45 fps at 4K) measured only the model forward
  pass. End-to-end rendering also decodes 4K, copies frames to/from CUDA, and
  encodes 4K60 H.264.
- During the 4K job, `ffmpeg`/libx264 was using about **870% CPU** while the
  GPU stayed near **1% utilization**. The pipeline was encoder-bound.
- `ffmpeg` cannot use NVENC in this pod because the container has
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility`; NVENC needs `video`
  (use `all` or `compute,utility,video`). `/dev/nvidia-caps` was empty.
- Measurements:
  - RIFE model only, 4K: ~20 fps full-res flow; ~45 fps UHD half-res flow.
  - Controlled 4K 2s interpolation-only: 13.2s for 119 frames (with
    libx264 medium).
  - Real 36s 4K 30→60 interpolation-only job: ~7.5 source fps / ~15 output
    fps, about 2.5 minutes; output was 2167 frames, 193 MB.
  - Combined 4K→8K interpolation + ONNX test: 0.5s clip in 22s; the 8K
    upscale dominates and is not a good candidate for 4K sources.

## Next steps

1. Restart the backend: `./scripts/run.sh`. It will re-probe encoders because
   the cache key now includes the driver-capabilities value.
2. For a large speedup, restart the pod with
   `NVIDIA_DRIVER_CAPABILITIES=all` (or `compute,utility,video`), then check
   `/api/encoders` for `h264_nvenc`/`hevc_nvenc`. No manual cache deletion
   should be needed.
3. Re-run the 4K clip as **No upscaling** + **RIFE v4.6 (CUDA)** to confirm the
   end-to-end rate. The displayed fps is source fps; output fps is 2x for a 2x
   render.
4. Avoid 4K→8K upscaling when only interpolation is wanted; the 8K ONNX pass
   is much more expensive than RIFE.
5. Once the backend is accepted, build the small frontend as requested.

## Resume commands

```bash
cd /home/ubuntu/projects/upscaler
.venv/bin/python -m pytest -q
./scripts/run.sh
```

Useful checks:

```bash
curl -s http://127.0.0.1:8000/api/interpolators
curl -s http://127.0.0.1:8000/api/encoders
nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv
pgrep -af "uvicorn|ffmpeg|Xvfb"
```
