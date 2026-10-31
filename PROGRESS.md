# RIFE CUDA progress — 2026-10-19

## 2026-10-31 performance pass (CUDA streaming)

The CUDA RIFE path was encoder/host-bound at ~5.4 source fps on a 10 s 4K30 →
60 fps render. Three fixes, all in `backend/engines/rife_torch.py` plus a
`pix_fmt` argument on `pipeline.open_frame_writer`:

1. **Pinned frame download.** A blocking pageable `.cpu()` of one 4K frame ran
   at ~1.5 GB/s (15–42 ms); a reusable pinned buffer + `non_blocking` copy runs
   at ~20 GB/s (~1–3 ms).
2. **GPU RGB→YUV420p.** ffmpeg's own rgb24→yuv420p conversion costs ~30 ms per
   4K frame and capped the stream at ~20 output fps. The engine now emits
   BT.601 limited-range yuv420p (matching ffmpeg's output within <1 level) and
   the writer takes `pix_fmt="yuv420p"`.
3. **Overlapped encode.** `_AsyncFrameWriter` feeds ffmpeg's stdin from a
   background thread so encoding runs while the next RIFE pass runs.

Measured on the same 10 s 4K30→60 clip (libx264 veryfast, CRF 20):

| stage | source fps | output fps | wall |
|---|---|---|---|
| baseline | 5.41 | 10.8 | 55.4 s |
| + pinned D2H | 6.05 | 12.1 | 49.6 s |
| + GPU YUV | 10.19 | 20.4 | 29.4 s |
| + overlap thread | 11.02 (12.3 steady) | 22.0 (24.6 steady) | 27.2 s |

PSNR vs the pre-change render: 44 dB. The remaining bottleneck is the producer:
RIFE forward ~30 ms, 4K decode/read ~19 ms, H2D ~10 ms per source frame. The
encoder now overlaps and is no longer the limit, so NVENC would give little
further gain for this path.

## Subtitle/audio output (2026-10-31)

- `encode.format` chooses the primary download: MP4 (default, browser) or MKV.
  The in-app player always uses the MP4.
- When the source has subtitles (or MKV is chosen), the job creates
  `output_subs.mkv` by stream-copying the MP4's video/audio plus the source's
  ASS/PGS subtitle and font streams (and chapters). The other container is the
  secondary download link.
- All audio tracks are preserved; Opus/FLAC/DTS are transcoded to AAC for MP4.
- No video/audio re-encode: the MKV mux is a copy pass.
- Large uploads use 8 MB chunks (`/api/upload/chunk`) so RunPod's HTTP proxy
  no longer returns 413 for big MP4/MKV files; chunks are assembled on disk.

## AMD / ROCm (checked 2026-10-31)

- The PyTorch RIFE path is not CUDA-specific: ROCm PyTorch implements the
  `torch.cuda` API, so `rife_torch.py` runs unchanged. `pin_memory`,
  `non_blocking` copies, `torch.cuda.synchronize`, `grid_sample` and
  `torch.backends.cudnn.benchmark` (MIOpen autotuning on current ROCm) are all
  supported.
- RX 9060 XT (gfx1200, RDNA 4) is officially supported by ROCm 10.1 on Linux
  and Windows 11. Linux:
  `pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1200]==2.14.0+rocm10.1.0" ...`
  Windows: install the gfx1200 wheel from
  `https://rocm.nightlies.amd.com/whl-multi-arch/` (newer/less mature stack).
- Vulkan ncnn engines work out of the box on AMD (Mesa RADV on Linux, the
  vendor driver on Windows).
- ONNX upscaler on Linux AMD is the gap: ORT >= 1.23 removed
  `ROCMExecutionProvider`; AMD now ships the MIGraphX plugin
  (`onnxruntime-ep-migraphx`), which `_GPU_PROVIDERS` does not register. Use
  the Vulkan upscaler or add MIGraphX support. On Windows, DirectML covers it.
- Performance is not measured here (no AMD hardware): the 9060 XT has roughly
  1/8 the FP32 compute and 1/3 the bandwidth of a 4090, so 4K RIFE is likely
  below real-time; 1080p should be comfortable. Benchmark on the card.
- Older ROCm builds reported only 8 GB of the 16 GB VRAM on this card (ROCm
  issue #6295); use a current build.


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
