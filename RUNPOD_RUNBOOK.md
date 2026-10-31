# RunPod deployment runbook — 4K Upscaler (RIFE CUDA interpolation)

This gets the upscaler running on a RunPod pod so you can interpolate anime
with **rife-v4.6 (CUDA)**. It assumes you already read the desktop workflow
and only need the container-specific steps.

## Why this fixes "can't use device 0"

- On RunPod your code already picks the GPU correctly: `rife_torch._load()`
  falls back to `cuda:0` when no `gpu_id` is given, and clamps any requested
  id to what `torch.cuda.device_count()` reports.
- The "cannot use device" error comes from a **mismatch between what
  `nvidia-smi` shows and what CUDA/torch can actually see**, controlled by
  `CUDA_VISIBLE_DEVICES` and the container's driver permissions.
- The app now surfaces that exact state: `GET /api/gpus` returns
  `cuda_visible_devices` and `torch_cuda_device_count`, and the RIFE engine's
  unavailability message includes `CUDA_VISIBLE_DEVICES=...`. Use those to
  diagnose on the pod.

## 1. Create the pod

- **GPU count: 1** (the app renders one job at a time; you don't need 8).
  Use a 1-GPU pod unless you specifically want parallel jobs later.
- **Template/container:** an NVIDIA CUDA image with PyTorch, e.g.
  `runpod/pytorch` or the RunPod "PyTorch" template matching your CUDA
  version. The setup script reuses the image's system PyTorch via the venv's
  `include-system-site-packages`, so do **not** pick an image without torch.
- **Container disk** ≥ 15 GB (RIFE weights + source videos + output).

### Set these env vars in the pod template (critical)

| Variable | Value | Why |
|---|---|---|
| `NVIDIA_DRIVER_CAPABILITIES` | `all` or `compute,utility,video` | Adds `video` so ffmpeg can use **NVENC**. Without it, 4K60 encoding falls back to libx264 at ~870% CPU and dominates render time. This is the single biggest speedup. |
| `UPSCALER_DATA_DIR` | `/workspace/upscaler-data` (default when `/workspace` exists) | Keeps job data in persistent disk. |

Leave `CUDA_VISIBLE_DEVICES` **unset** for a single-GPU pod so
`cuda:0 == the only GPU`.

## 2. Get the repo onto the pod

```bash
git clone <your-repo-url> /workspace/upscaler && cd /workspace/upscaler
# or copy it up with scp/rsync if it's not pushed
```

## 3. Setup

```bash
./scripts/setup.sh          # installs ffmpeg/Vulkan, .venv, deps, RIFE weights,
                            # and NVIDIA binary engines when a GPU is detected
source .venv/bin/activate
```

If `setup.sh` skips the ncnn/downloads because it didn't see the GPU, verify:
`nvidia-smi` shows your card and `echo $CUDA_VISIBLE_DEVICES`. The **CUDA RIFE
path does not need Vulkan or Xvfb**, so that's your primary engine.

## 4. Start the server

```bash
./scripts/run.sh            # binds 0.0.0.0:8000 behind the pod's proxy
```

The RunPod TCP proxy (port 8000) will forward to this. See the pod's
"Connect" → "TCP" tab for the public URL.

## 5. Verify on the pod

```bash
curl -s http://127.0.0.1:8000/api/gpus
# Expect: "cuda_visible_devices": null (or a single id), "torch_cuda_device_count": 1

curl -s http://127.0.0.1:8000/api/interpolators   # rife-v4.6 should be available
curl -s http://127.0.0.1:8000/api/encoders        # want h264_nvenc / hevc_nvenc

nvidia-smi --query-gpu=utilization.gpu --format=csv   # should be busy during a job
```

If `/api/interpolators` says `rife-v4.6` is unavailable, the reason now states
the exact `CUDA_VISIBLE_DEVICES` / zero-device condition — paste that here.

## 6. Run a job

- Engine: **No upscaling** (or your upscaler) + Interpolation: **RIFE v4.6 (CUDA)**,
  factor 2, mode **smooth**.
- GPU picker: leave **Auto** (→ `cuda:0`).
- The progress line counts **source frames**; a 2x job at 7.5 source fps is
  really ~15 output fps.

## Diagnostics cheat-sheet for "can't use device 0"

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi` shows 8 GPUs but sqlite `device_count` is 1 | Multiple GPUs but only one visible to CUDA | Use `CUDA_VISIBLE_DEVICES<id>` or just pick a 1-GPU pod |
| `torch_cuda_device_count` is `0` while `nvidia-smi` shows cards | Driver/cgroup caps hiding CUDA | Ensure pod exposes a real CUDA GPU; check driver version |
| `/api/encoders` lacks `h264_nvenc` | `NVIDIA_DRIVER_CAPABILITIES` missing `video` | Set `all` or `compute,utility,video`, restart pod |
| `cuda:N` works on desktop but not pod | Physical vs logical id mismatch | Rely on `Auto` (cuda:0) and the diagnostics above |

## Notes

- Multi-GPU: to run N jobs in parallel, launch one app instance per GPU by
  setting `CUDA_VISIBLE_DEVICES` to a single id per instance, and expose each
  on its own port. The current app architecture is **single-render-per-worker**,
  so parallel means multiple instances, not 8 GPUs in one app.
