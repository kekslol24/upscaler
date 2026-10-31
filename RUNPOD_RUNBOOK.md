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

- **GPU count:** 1 is enough for serial renders; the app now runs one render
  per GPU, so add GPUs only if you want parallel episodes. CPU encode and disk
  are shared across them.
- **Template/container:** an NVIDIA CUDA image with PyTorch, e.g.
  `runpod/pytorch` or the RunPod "PyTorch" template matching your CUDA
  version. The setup script reuses the image's system PyTorch via the venv's
  `include-system-site-packages`, so do **not** pick an image without torch.
- **Container disk** ≥ 15 GB (RIFE weights + source videos + output).

### Set these env vars in the pod template (critical)

| Variable | Value | Why |
|---|---|---|
| `NVIDIA_DRIVER_CAPABILITIES` | `all` or `compute,utility,video` | Adds `video` so ffmpeg can use **NVENC**. Without it, 4K60 encoding falls back to libx264. After the GPU YUV + background-writer changes the CPU encoder is no longer the dominant bottleneck, but NVENC still helps. |
| `UPSCALER_DATA_DIR` | `/workspace/upscaler-data` (default when `/workspace` exists) | Keeps job data in persistent disk. |

Leave `CUDA_VISIBLE_DEVICES` **unset** so the app sees every GPU; the worker
pool runs one render per GPU. Set it only to restrict the app to a subset.

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

## 6b. Performance expectations

Measured on the RTX 4090 pod, 10 s 4K30 → 60 fps, libx264 `veryfast`, CRF 20:

| stage | source fps | output fps | wall |
|---|---|---|---|
| original CUDA path | 5.4 | 10.8 | 55 s |
| + pinned GPU→CPU copy | 6.1 | 12.1 | 50 s |
| + GPU RGB→YUV420p | 10.2 | 20.4 | 29 s |
| + overlapped encode | 11.0 (12.3 steady) | 22.0 (24.6 steady) | 27 s |

The RIFE forward pass alone is ~23 ms/frame (44 fps) at 4K in UHD half-res
mode. The pipeline is now **producer-bound** (4K decode/pipe ~19 ms, H2D
~10 ms); the encoder runs on a background thread. A 30–50% `nvidia-smi`
reading is normal — GPU utilization is not the metric to optimize. If NVENC is
unavailable, the CPU libx264 path still does ~52–63 fps at 4K60 because
RGB→YUV happens on the GPU.

The same PyTorch engine also runs on AMD ROCm (it implements the `torch.cuda`
API); this runbook's NVIDIA-specific bits are the NVENC caps and CUDA wheels.

## 6c. Two or more GPUs

The app starts one worker thread per detected GPU automatically; no extra
config is needed.

- Create the pod with N GPUs, a PyTorch template, and
  `NVIDIA_DRIVER_CAPABILITIES=all` (or `compute,utility,video`).
- Clone, setup and run exactly as above. `curl -s
  http://127.0.0.1:8000/api/gpus` should list all N GPUs.
- In the UI, leave **GPU** on **Auto**, select multiple episodes, and click
  **Render all uploaded**. Auto gives each job to the least-loaded GPU and
  queues extras behind busy ones. Pick a GPU explicitly to pin one episode.
- `UPSCALER_WORKERS=N` caps concurrent workers (default: all GPUs) if the
  pod's CPU or disk cannot feed them. One concurrent 4K encode uses ~4–5 CPU
  cores, and disk usage scales with the number of outputs.
- NVENC needs the `video` driver capability on every GPU; without it the
  jobs fall back to libx264 and compete for CPU. Check `/api/encoders`.
- The worker count is fixed at startup; restart the app if you add/remove
  GPUs.
- Without a persistent `/workspace`, job data and outputs are lost on pod
  restart. The repo, venv and engines are recreated by clone + setup.

## Diagnostics cheat-sheet for "can't use device 0"

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi` shows 8 GPUs but sqlite `device_count` is 1 | Multiple GPUs but only one visible to CUDA | Use `CUDA_VISIBLE_DEVICES<id>` or just pick a 1-GPU pod |
| `torch_cuda_device_count` is `0` while `nvidia-smi` shows cards | Driver/cgroup caps hiding CUDA | Ensure pod exposes a real CUDA GPU; check driver version |
| `/api/encoders` lacks `h264_nvenc` | `NVIDIA_DRIVER_CAPABILITIES` missing `video`, or `/dev/nvidia-caps` has no devices | Set `all` or `compute,utility,video` and recreate the pod (the runtime injects the cap devices at container start; exporting the env var inside a running container is not enough) |
| `cuda:N` works on desktop but not pod | Physical vs logical id mismatch | Rely on `Auto` (cuda:0) and the diagnostics above |

## Notes

- Multi-GPU: the app starts one worker per GPU automatically. "Auto" assigns
  the least-loaded GPU; pin an episode by choosing a GPU in the form.
  `UPSCALER_WORKERS=N` caps concurrency (default: every detected GPU). One
  4K H.264 encode uses ~4–5 CPU cores, so size the pod's CPU and disk for the
  number of concurrent episodes. A 1-GPU pod still works as before.
