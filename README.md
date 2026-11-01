# 4K Upscaler

Local video upscaler with a pluggable model "harness". Upload a short clip,
pick an engine, get a before/after slider.

Three upscaling paths are available:

- **Anime4K v4 (default)**: GPU GLSL shader chain via ffmpeg/libplacebo.
  Much cheaper than the neural models; it sharpens and restores line art
  rather than inventing detail. The preset picks it automatically.
- **DirectML/CUDA**: ONNX Runtime, whole-frame inference, frames streamed
  straight between ffmpeg decoder and encoder. Several times faster than
  the Vulkan build and needs no scratch disk space. Output is identical
  to within one 8-bit level.
- **Vulkan (ncnn)**: the portable prebuilt binary; works on Linux too.

Frame interpolation has the same choice: the PyTorch RIFE engine (CUDA on
NVIDIA, ROCm on AMD) streams frames with no PNG scratch files, while the
Vulkan build remains available as a portable fallback.

## Setup

On Windows the ONNX engine uses DirectML; on Linux with an NVIDIA GPU it uses
CUDA (installed by `scripts/setup.sh`). The Vulkan engines cover AMD/Intel
GPUs and run on whatever Vulkan can see, so run them natively on the OS that
owns the GPU. WSL2 works too, but Ubuntu's stock Mesa has no Vulkan-on-D3D12
(Dozen) driver, so `vulkaninfo` only sees the `llvmpipe` CPU fallback until
you install it — `scripts/setup_amd.sh` does that via the kisak-mesa PPA.

AMD GPUs are supported on three levels: the Vulkan engines work out of the
box, the PyTorch RIFE path also runs on ROCm (PyTorch exposes the CUDA API on
ROCm, so the same engine works), and the Anime4K shader engine uses Vulkan.
On WSL2 the Anime4K engine prefers a Windows `ffmpeg.exe` with libplacebo
(found via `where ffmpeg`) because Mesa's Dozen Vulkan path is much slower;
on native Linux/Windows it uses the normal ffmpeg. See
[AMD GPUs](#amd-gpus-rocm--vulkan).

### Windows

From a PowerShell prompt in the project folder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

This installs `ffmpeg` via winget, creates a Python venv, installs the
Python deps, and downloads the Windows build of the Real-ESRGAN ncnn-vulkan
engine. Python 3.10+ must already be installed (`winget install
Python.Python.3.12`). If ffmpeg was just installed, open a new terminal
before starting the server so it is on PATH.

### Linux (Arch, Ubuntu/Debian)

```bash
./scripts/setup.sh
```

This installs `ffmpeg`/Vulkan via pacman or apt, creates a Python venv,
installs Python deps, downloads the RIFE v4.6 PyTorch weights, and downloads
the Real-ESRGAN ncnn-vulkan engine binary (runs on Intel/AMD/NVIDIA GPUs via
Vulkan — no CUDA needed). When an NVIDIA GPU is present it also installs
`onnxruntime-gpu`, so the fast ONNX engine runs on CUDA and a GPU smoke test
is performed. If the image already ships PyTorch in its system interpreter,
setup enables system site packages in `.venv` instead of downloading a second
torch wheel. Pass `--with-vulkan` or `--skip-vulkan` to override the automatic
Vulkan download decision. On AMD, install a ROCm PyTorch build (below) and the
Vulkan engines work out of the box.

### AMD GPUs (ROCm / Vulkan)

The quickest path on Ubuntu/WSL2 is:

```bash
./scripts/setup_amd.sh          # or ./setup_amd.sh from the repo root
```

It installs Mesa's Dozen driver on WSL2, runs the normal setup with the Vulkan
engines, optionally installs the gfx1200 ROCm PyTorch wheel (`--rocm`), checks
which engines are usable, and starts the server. `--no-start` sets up only,
`--vulkan` never touches ROCm, and `--yes` accepts all prompts.

**Vulkan (works everywhere).** `scripts/setup.sh` installs the Mesa Vulkan
driver on non-NVIDIA Linux and downloads the ncnn Real-ESRGAN/RIFE binaries
when `vulkaninfo` sees a discrete GPU. On Windows, `setup.ps1` downloads the
same binaries and `onnxruntime-directml` for the ONNX engine. This is the
zero-extra-install path for AMD.

**ROCm (fast PyTorch RIFE).** The `rife-v4.6` engine is not CUDA-specific: it
uses the `torch.cuda` API, which ROCm PyTorch implements. Install a ROCm
PyTorch build for your GPU before or after `setup.sh`:

```bash
# Linux, RX 9060 XT (gfx1200), ROCm 10.1
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "torch[device-gfx1200]==2.14.0+rocm10.1.0" \
  "torchvision[device-gfx1200]==0.29.0a0+rocm10.1.0" \
  "torchaudio==2.11.0.3+rocm10.1.0"
```

On Windows 11, install the ROCm PyTorch wheel for gfx1200 from AMD's
multi-arch index into `.venv` (ROCm 10.1 / PyTorch 2.14; the Windows ROCm
stack is newer and less mature than Linux):

```powershell
.venv\Scripts\python.exe -m pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ `
  "torch[device-gfx1200]" "torchvision[device-gfx1200]" torchaudio
```

Then restart and check `GET /api/interpolators`; `rife-v4.6` should be
available. If it is not, the reason string says whether PyTorch CUDA/ROCm is
unavailable or the weights are missing.

Notes for the RX 9060 XT (gfx1200, RDNA 4):

- Officially supported by ROCm 10.1 on Linux and Windows 11. Use a current
  amdgpu/Adrenalin driver and a current ROCm/PyTorch build: older builds could
  report only ~8 GB of the 16 GB VRAM (ROCm issue #6295) and segfault on
  larger allocations.
- Expect it to be much slower than an RTX 4090: roughly 1/8 the FP32 compute
  and 1/3 the memory bandwidth. 4K RIFE is likely below real-time; 1080p is
  comfortable. Benchmark on the actual card with `scripts/bench.py`.
- The ONNX upscaler works on Linux AMD too. `setup_amd.sh` installs
  `onnxruntime-rocm==1.22.2.post3` against the gfx1200 ROCm 10.1 PyTorch stack
  (`post1` for native ROCm 6 installs), and the engine selects
  `ROCMExecutionProvider`. `MIGraphXExecutionProvider` is also accepted when a
  newer ORT build provides it. Start the server with `./scripts/run.sh` (or
  `python launcher.py`) so the ROCm libraries bundled in site-packages and the
  `librocm_smi64.so.1` shim are on the loader path; a bare
  `uvicorn backend.main:app` will report the ONNX engine as unavailable.
  On Windows, DirectML already covers AMD for the ONNX engine.
- Hardware encoding: Windows AMD can use AMF (`h264_amf`/`hevc_amf`) if the
  ffmpeg build supports it; Linux AMD has no VAAPI candidate in the encoder
  list, so it falls back to libx264. The GPU YUV conversion and background
  writer make that CPU path much less painful.

### RunPod / headless GPU containers

On a RunPod pod using a **PyTorch** template:

```bash
./scripts/setup.sh
./scripts/run.sh
```

`setup.sh` detects the NVIDIA GPU, installs ffmpeg and the venv, and installs
a pinned `onnxruntime-gpu` matching the image's CUDA/cuDNN (CUDA 12.0–12.7,
12.8+, and 13.x each need a different build; the CUDA 13 default on PyPI does
not load on CUDA 12 images). It reuses the image's PyTorch installation for the
CUDA RIFE engine and downloads the RIFE v4.6 weights. It also downloads the
ncnn/Vulkan binaries when Vulkan sees a real GPU; on headless hosts it probes
through a temporary virtual X display, because NVIDIA's Vulkan ICD can refuse
to initialize without one. It ends with a real x2 inference that must run on
`CUDAExecutionProvider` — a CUDA/cuDNN mismatch fails at setup time instead of
silently falling back to the CPU, and the app itself now refuses to fall back
too.

On a 4K source, prefer **No upscaling** when you only need interpolation:
upscaling 4K to 8K is much more expensive than generating the extra frames.
If ffmpeg falls back to `libx264` (no NVENC), see the performance notes below:
the GPU YUV conversion and background writer mean the CPU encoder is no longer
the dominant bottleneck, but NVENC still helps. NVENC needs the container's
`video` driver capability (RunPod templates sometimes expose only
`compute,utility`).

#### Reading the interpolate FPS

The progress line for interpolation counts **source** frames. A 2x render
therefore produces about twice the displayed fps (e.g. `7.5 fps` in the UI
means ~15 output fps).

On the RTX 4090 pod, a 10 s 4K30 → 60 fps interpolation-only render (libx264
`veryfast`, CRF 20) measured:

| stage | source fps | output fps | wall |
|---|---|---|---|
| original CUDA path | 5.4 | 10.8 | 55 s |
| + pinned GPU→CPU copy | 6.1 | 12.1 | 50 s |
| + GPU RGB→YUV420p | 10.2 | 20.4 | 29 s |
| + overlapped encode | 11.0 (12.3 steady) | 22.0 (24.6 steady) | 27 s |

The RIFE forward pass alone is ~23 ms/frame (44 fps) at 4K in UHD
half-resolution mode. The remaining cost is the producer: 4K decode/pipe read
(~19 ms), H2D (~10 ms) and the forward pass. The encoder now runs on a
background thread, so `nvidia-smi` may show only 30–50% GPU utilization — the
GPU idles during the CPU decode/copy stages, and utilization is not the metric
to optimize.

NVENC still helps when available, but it is no longer the dominant bottleneck:
with GPU YUV the CPU libx264 path already does ~52–63 fps at 4K60. NVENC needs
the container's `video` driver capability (`NVIDIA_DRIVER_CAPABILITIES=all` or
`compute,utility,video`); the encoder cache key includes the capability value,
so a restart automatically re-probes and picks up `h264_nvenc`/`hevc_nvenc`.

`run.sh` listens on `0.0.0.0:8000` and starts a virtual X display automatically
when `DISPLAY` is unset, so the Vulkan engines and RIFE work in the headless
container. When `/workspace` exists, it keeps job
data in `/workspace/upscaler-data` (survives pod restarts). Keep the project
itself on `/workspace` too, so the venv and downloaded engines persist. Expose
**8000** as an HTTP port in the pod settings, then open the pod's proxy URL.
Override with `UVICORN_HOST`/`UVICORN_PORT`, and set `UPSCALER_DATA_DIR` if
the volume is mounted elsewhere.

If the pod template is not CUDA 12.x, pick one that is (the ORT GPU wheel
needs the matching CUDA/cuDNN), or use an image that ships PyTorch.

## Run

Windows (no activation needed, so it works regardless of execution policy):

```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

If you prefer `.venv\Scripts\activate`, allow local scripts once for your
user first: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Linux:

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000. Picking a file uploads it immediately; then:

1. choose a **preset** (or tweak engine / scale / filters / output by hand),
2. hit **Preview clip** to render only a few seconds from a chosen timestamp
   and compare settings in seconds,
3. hit **Render full video** once you like the result.

Full 4K renders on integrated graphics are slow — that's expected. A
running render can be **paused** and **resumed** between frames, or cancelled;
pausing parks the worker without closing the encoder, so it keeps its current
progress. Engines that run as one external process (Anime4K, the ncnn/Vulkan
engines) show no Pause button; cancel those instead. A paused render does not
survive a server restart (the restart marks it interrupted), so resume it
before restarting the server. Finished jobs stay listed in **Recent jobs**
across server restarts (job state lives in `data/jobs/jobs.sqlite3`) and can
be re-used as a source or viewed again.

### Parallel renders (one episode per GPU)

`backend/jobs.py` starts one worker per detected GPU. A new job with
**GPU: Auto** goes to the least-loaded GPU; pick a specific GPU in the form
to pin an episode. You can upload and start the next episode while another is
rendering — **Recent jobs** refreshes every few seconds and **Watch** switches
the progress view to any running job.

The file picker accepts multiple files. Upload a batch, set the options once,
then **Render all uploaded**; with **GPU: Auto** the backend spreads them
across the GPUs and queues extra episodes as GPUs free up.

`UPSCALER_WORKERS=N` caps the number of concurrent GPU workers (default:
every detected GPU). CPU encoding and disk are shared: one 4K H.264 encode
uses roughly 4–5 cores, so N concurrent 4K renders need about N× the CPU and
N× the disk. A 1-GPU machine behaves exactly as before.

### Filters and output

- **Deinterlace** (`bwdif`) and **denoise** (`hqdn3d`, three strengths) run
  on the decoded source *before* the model, so it doesn't sharpen
  compression noise; **sharpen** (`unsharp`) runs on the upscaled output.
- **Codec/encoder**: H.264 or HEVC, auto-picking a working GPU encoder
  (NVENC / AMF / Quick Sync / VideoToolbox) with a CPU fallback. Each encoder
  is test-run once at startup so only ones that actually work are offered.
  Quality is CRF-style (18 default, lower = better).
- **Container**: MP4 (browser-playable) or MKV (keeps ASS/PGS subtitles and
  fonts). This is the primary download; the other container is offered as a
  secondary link when it exists.
- Audio is stream-copied when it is MP4-friendly (AAC/MP3/AC3) and
  transcoded to AAC otherwise (Opus/FLAC/DTS from MKV sources).
- The before/after view gets a browser-playable H.264 proxy of the source
  window when the upload itself isn't playable (10-bit HEVC in MKV, say).

### Output container, subtitles and audio

- The **Container** control chooses the primary download: **MP4** (browser) or
  **MKV** (keeps ASS/PGS subtitles and fonts). The in-app before/after player
  always uses the browser-playable MP4.
- When the source has subtitles (or MKV is chosen), the job also creates an
  MKV by stream-copying the MP4's video/audio plus the source's subtitle and
  font streams and the chapters. Video/audio are not re-encoded, so ASS styling
  and PGS (signs/songs) survive.
- The chosen container is the main download; the other one is offered as a
  secondary link when it exists.
- All source audio tracks are kept. MP4-safe codecs (AAC/MP3/AC3/EAC3) are
  copied; anything else (Opus/FLAC/DTS) is transcoded to AAC 192k. Track order
  is preserved and the first track is marked default.
- Subtitles are not burned in. The MP4 itself has no subtitle stream (MP4
  cannot carry ASS/PGS); use the MKV download to keep them.

### Large uploads and input formats

MP4, MKV, MOV and anything else ffmpeg can decode are accepted as input. Large
files are uploaded in **8 MB chunks** (`/api/upload/chunk`), so a reverse
proxy's per-request body limit — including RunPod's HTTP proxy, which returns
**413 Payload Too Large** for big single POSTs — no longer applies. The chunks
are assembled on disk, not held in RAM.

Presets live in `backend/presets.py`; a preset is just a bundle of form
values with an engine preference list, so adding one is a few lines.

### Frame interpolation (RIFE)

RIFE v4.6 adds a frame-rate stage:

- **Smooth motion**: 2x/3x/4x the frame rate (24 → 48/72/96 fps), same
  duration, audio kept.
- **Slow motion**: the same extra frames played at the original rate, so the
  clip runs 2–4x slower; audio is dropped.
- Pick **No upscaling** as the engine to interpolate only.

The default **RIFE v4.6 (CUDA/ROCm)** engine runs the network through PyTorch
and streams frames directly into ffmpeg. It converts RGB→YUV420p on the GPU
(BT.601 limited range, matching ffmpeg's own conversion, so colors are
unchanged), downloads frames through a reusable pinned host buffer, and
encodes on a background thread while the next frame is interpolated. On an
RTX 4090 a 4K pair takes ~23 ms in UHD half-resolution mode; end-to-end
4K30→60 interpolation runs at ~11 source fps (~22 output fps) with libx264,
up from ~5.4 source fps before these fixes. Interpolation-only renders bypass
the chunked pipeline completely.

When both upscaling and interpolation are selected, the stage order is
automatic: a streaming upscaler (ONNX) interpolates at source resolution
first and then upscales; a directory-mode ncnn/Vulkan upscaler upscales
first and then interpolates. That keeps RIFE off 4K frames on the fast path
and avoids doubling the slow ncnn upscaler's frame count.

The Vulkan fallback (`rife-ncnn-vulkan`, downloaded into `engines_bin/rife/`)
still exists for machines without PyTorch/CUDA/ROCm. Renders that use a
directory-mode stage (the Vulkan interpolator or an ncnn upscaler) go through
the **chunked pipeline**: the source is decoded once and cut into 5-second
chunks of PNG frames, each chunk is upscaled → interpolated → encoded to a
segment and its frames deleted, and the segments are concatenated losslessly
at the end. Disk use stays bounded, and chunks overlap by one frame so the
interpolator sees every frame pair.

### Engines

| Engine | Runs on | Good for |
|---|---|---|
| Anime4K v4 | ffmpeg/libplacebo, streaming | fastest anime line enhancer; on WSL uses the Windows ffmpeg when available |
| Real-ESRGAN anime video (DirectML/CUDA) | ONNX Runtime, streaming | anime, fastest on NVIDIA/Windows AMD |
| Real-ESRGAN anime video (Vulkan) | ncnn binary | anime, any GPU/OS |
| Real-ESRGAN anime, max detail | ncnn binary | anime stills model, crispest lines |
| Real-ESRGAN live action / general | ncnn binary | real footage, 4x only, slow |
| RIFE v4.6 (CUDA/ROCm) | PyTorch on CUDA or ROCm, streaming | frame interpolation, fastest |
| RIFE v4.6 (Vulkan) | ncnn binary | frame interpolation, portable fallback |

## Desktop build (Windows .exe / installer)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1      # once
powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
```

`build_exe.ps1` runs PyInstaller with `upscaler.spec`, copies `engines_bin/`
and `models/` next to the exe, bundles `ffmpeg.exe`/`ffprobe.exe` into
`bin/`, smoke-tests that the app starts, and — if Inno Setup 6 is installed
(`winget install JRSoftware.InnoSetup`) — compiles `installer/upscaler.iss`
into `dist\UpscalerSetup.exe`. Without Inno Setup you still get
`dist\Upscaler\`, a portable folder: run `Upscaler.exe` and it opens the
browser on a free local port.

Why it launches quickly: it is a **one-folder** build (a one-file exe would
unpack ~300 MB to a temp folder on every start), encoder detection is cached
in the data folder, and the ONNX runtime only loads when a job starts. Job
data and logs live in `%LOCALAPPDATA%\Upscaler` (override with
`UPSCALER_DATA_DIR`), so reinstalling or updating never touches renders.

`python launcher.py` runs the same entry point from a checkout.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers the ffmpeg pipeline (clip windows, filters, audio
transcode, encoders), the exported ONNX models (output shape, no black
rows, band tiling equivalence) and the job manager + API end to end,
rendering a tiny clip on the CPU ONNX provider. Set `UPSCALER_ALLOW_CPU=1`
to allow the ONNX engine on the CPU provider outside tests (20 s per 1080p
frame — useful for checking correctness, not for real renders).

## Regenerating the ONNX models

`models/*.onnx` are exported from the official `realesr-animevideov3.pth`
by `scripts/export_onnx.py` (needs torch, onnx and onnxruntime; CPU is
fine). You only need this if you want to change the export.

## Adding a new model engine

Implement `backend/engines/base.py::UpscaleEngine` in a new file under
`backend/engines/`, then register an instance in `backend/engines/__init__.py`.
The rest of the pipeline (frame extraction, job queue, API, frontend) is
engine-agnostic — any engine that turns a directory of frame PNGs into a
directory of upscaled frame PNGs works. Set `streaming = True` and
implement `upscale_video()` instead to get raw frames over pipes with no
intermediate files (see `realesrgan_onnx.py`).
