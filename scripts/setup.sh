#!/usr/bin/env bash
# Installs system deps, sets up the Python venv, enables GPU inference and
# downloads the Vulkan engine binaries when the machine can actually run them.
#
# Works on Arch, Ubuntu/Debian (incl. WSL2) and headless GPU containers such
# as RunPod's PyTorch image. Safe to re-run.
#
# Usage:
#   ./scripts/setup.sh                 # auto: CUDA if NVIDIA, Vulkan if usable
#   ./scripts/setup.sh --with-vulkan   # also download the ncnn/RIFE binaries
#   ./scripts/setup.sh --skip-vulkan   # never download them
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WITH_VULKAN=auto
for arg in "$@"; do
  case "$arg" in
    --with-vulkan) WITH_VULKAN=yes ;;
    --skip-vulkan) WITH_VULKAN=no ;;
    -h|--help)
      echo "usage: $0 [--with-vulkan|--skip-vulkan]"
      exit 0
      ;;
    *)
      echo "unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "error: not running as root and sudo is not installed; re-run as root." >&2
    exit 1
  fi
fi

PYTHON=python3
command -v python3 >/dev/null 2>&1 || PYTHON=python

IS_WSL=false
if grep -qi microsoft /proc/version 2>/dev/null; then
  IS_WSL=true
fi

HAS_NVIDIA=false
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAS_NVIDIA=true
fi

if $HAS_NVIDIA; then
  GPU_LINE="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  echo "== NVIDIA GPU: ${GPU_LINE:-detected} =="
fi

echo "== System packages (ffmpeg + unzip + Vulkan loader/tools) =="
if command -v pacman >/dev/null 2>&1; then
  $SUDO pacman -S --needed --noconfirm ffmpeg unzip curl python python-pip vulkan-icd-loader vulkan-tools xorg-server-xvfb xorg-xauth
  if ! $HAS_NVIDIA; then
    $SUDO pacman -S --needed --noconfirm vulkan-radeon || true
  fi
elif command -v apt-get >/dev/null 2>&1; then
  # Note: on WSL2 Ubuntu's Mesa does not ship the Dozen (Vulkan-on-D3D12)
  # driver, so Vulkan only sees the llvmpipe CPU fallback there. For GPU
  # acceleration on a Windows machine run the project natively on Windows
  # with scripts/setup.ps1 instead.
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get update
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ffmpeg unzip curl python3-venv python3-pip xvfb xauth
  # libvulkan1 + vulkan-tools let the ncnn engines run where the driver
  # provides a Vulkan ICD; they are optional, so a missing repo entry must
  # not abort setup on minimal container images.
  if $HAS_NVIDIA; then
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y \
      libvulkan1 vulkan-tools || true
  else
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y \
      libvulkan1 mesa-vulkan-drivers vulkan-tools || true
  fi
else
  echo "no pacman/apt found — install ffmpeg, unzip, curl and your GPU's"
  echo "runtime/Vulkan driver manually, then re-run this script."
fi

if $IS_WSL && [[ "$ROOT_DIR" == /mnt/* ]]; then
  echo
  echo "note: project lives on a Windows drive ($ROOT_DIR). File I/O through"
  echo "/mnt/<drive> is slow in WSL — pip install and frame extraction will take"
  echo "longer than on the Linux filesystem (e.g. ~/upscaler). It still works."
fi

echo "== Python venv =="
# A half-created .venv (e.g. made before python3-venv was installed) has no
# pip launcher, so a bare `pip` falls through to the system pip and fails with
# "externally-managed-environment". Recreate it in that case and always call
# pip via the venv's python.
if [ ! -x .venv/bin/python ] || [ ! -x .venv/bin/pip ] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m ensurepip --upgrade >/dev/null 2>&1 || true
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# CUDA images (RunPod etc.) ship PyTorch in the system interpreter. Let the
# venv see it instead of downloading a second multi-GB torch wheel.
if "$PYTHON" -c 'import torch' >/dev/null 2>&1 && ! .venv/bin/python -c 'import torch' >/dev/null 2>&1; then
  echo "== Using the image's PyTorch (system site packages) =="
  sed -i 's/^include-system-site-packages = false/include-system-site-packages = true/' .venv/pyvenv.cfg
  if ! .venv/bin/python -c 'import torch' >/dev/null 2>&1; then
    echo "warning: system PyTorch is still not importable from .venv" >&2
  fi
fi

echo "== RIFE PyTorch weights =="
RIFE_MODEL_DIR="$ROOT_DIR/models/rife-v4.6"
if [ -f "$RIFE_MODEL_DIR/flownet.pkl" ]; then
  echo "already installed at $RIFE_MODEL_DIR"
else
  mkdir -p "$RIFE_MODEL_DIR"
  TMP_ZIP="$(mktemp)"
  if curl -fsSL --retry 3 -o "$TMP_ZIP" \
    "https://huggingface.co/Bash2X/RIFE-Models/resolve/main/RIFE_v4.6.zip" \
    && unzip -o -q "$TMP_ZIP" flownet.pkl -d "$RIFE_MODEL_DIR"; then
    echo "installed to $RIFE_MODEL_DIR"
  else
    echo "warning: could not install RIFE v4.6 weights; the CUDA interpolator" >&2
    echo "will be unavailable until $RIFE_MODEL_DIR/flownet.pkl exists." >&2
  fi
  rm -f "$TMP_ZIP"
fi

echo "== ONNX Runtime GPU provider =="
if $HAS_NVIDIA; then
  # The ORT wheel has to match the image's CUDA and cuDNN runtime. Since
  # 1.27 the PyPI default is a CUDA 13 build, so blindly installing the
  # latest breaks every CUDA 12 image (libcublasLt.so.13 not found).
  CUDA_VERSION="$("$PYTHON" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
  CUDNN_VERSION="$("$PYTHON" -c 'import torch; print(torch.backends.cudnn.version() or "")' 2>/dev/null || true)"
  CUDA_MAJOR="${CUDA_VERSION%%.*}"
  CUDA_MINOR="${CUDA_VERSION#*.}"
  CUDA_MINOR="${CUDA_MINOR%%.*}"
  CUDNN_MAJOR=9
  if [[ "$CUDNN_VERSION" =~ ^[0-9]+$ ]]; then
    CUDNN_MAJOR=$((CUDNN_VERSION / 10000))
  fi

  ORT_SPEC="onnxruntime-gpu==1.20.2"
  NUMPY_SPEC=""
  if [ "$CUDA_MAJOR" = 11 ]; then
    # CUDA 11.8 is only used by PyTorch <= 2.3 images (Python <= 3.11).
    ORT_SPEC="onnxruntime-gpu==1.17.1"
    NUMPY_SPEC="numpy<2"
  elif [ "$CUDNN_MAJOR" -lt 9 ]; then
    # cuDNN 8 (PyTorch <= 2.3): last cuDNN 8 wheel on PyPI.
    ORT_SPEC="onnxruntime-gpu==1.18.0"
    NUMPY_SPEC="numpy<2"
  elif [ "$CUDA_MAJOR" -ge 13 ]; then
    ORT_SPEC="onnxruntime-gpu>=1.27,<2"
  elif [ "$CUDA_MAJOR" = 12 ] && [ "${CUDA_MINOR:-0}" -ge 8 ]; then
    # Built with CUDA 12.8; supports Python 3.10-3.13.
    ORT_SPEC="onnxruntime-gpu==1.21.1"
  fi
  echo "CUDA ${CUDA_VERSION:-unknown}, cuDNN ${CUDNN_VERSION:-unknown} -> $ORT_SPEC"
  specs=("$ORT_SPEC")
  [ -n "$NUMPY_SPEC" ] && specs=("$NUMPY_SPEC" "$ORT_SPEC")
  .venv/bin/python -m pip install "${specs[@]}"
  .venv/bin/python - <<'PY'
import onnxruntime as ort
print("onnxruntime", ort.__version__)
PY
else
  echo "no NVIDIA GPU found — the ONNX engine will use the CPU provider"
  echo "(set UPSCALER_ALLOW_CPU=1 to allow it, useful only for tests)."
fi

vulkan_summary() {
  if [ -n "${DISPLAY:-}" ] || ! command -v xvfb-run >/dev/null 2>&1; then
    vulkaninfo --summary 2>/dev/null || true
  else
    # NVIDIA's Vulkan ICD can refuse to initialize without an X display,
    # even on a headless host. A virtual display is enough to enumerate it.
    xvfb-run -a vulkaninfo --summary 2>/dev/null || true
  fi
}

vulkan_gpu_detected() {
  command -v vulkaninfo >/dev/null 2>&1 || return 1
  vulkan_summary \
    | grep -Eq 'deviceType[[:space:]]*=[[:space:]]*PHYSICAL_DEVICE_TYPE_(DISCRETE_GPU|INTEGRATED_GPU)'
}

WANT_VULKAN=false
if [ "$WITH_VULKAN" = yes ]; then
  WANT_VULKAN=true
elif [ "$WITH_VULKAN" = auto ] && vulkan_gpu_detected; then
  WANT_VULKAN=true
fi

if $WANT_VULKAN; then
  echo "== Real-ESRGAN ncnn-vulkan engine binary =="
  ENGINE_DIR="$ROOT_DIR/engines_bin/realesrgan"
  if [ -x "$ENGINE_DIR/realesrgan-ncnn-vulkan" ]; then
    echo "already installed at $ENGINE_DIR"
  else
    mkdir -p "$ENGINE_DIR"
    TMP_ZIP="$(mktemp)"
    curl -fsSL --retry 3 -o "$TMP_ZIP" \
      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
    unzip -o -q "$TMP_ZIP" -d "$ENGINE_DIR"
    chmod +x "$ENGINE_DIR/realesrgan-ncnn-vulkan"
    rm -f "$TMP_ZIP"
    echo "installed to $ENGINE_DIR"
  fi

  echo "== RIFE ncnn-vulkan frame interpolation binary =="
  RIFE_DIR="$ROOT_DIR/engines_bin/rife"
  if [ -x "$RIFE_DIR/rife-ncnn-vulkan" ] && [ -d "$RIFE_DIR/rife-v4.6" ]; then
    echo "already installed at $RIFE_DIR"
  else
    mkdir -p "$RIFE_DIR"
    TMP_ZIP="$(mktemp)"
    TMP_DIR="$(mktemp -d)"
    # ~430 MB: the release bundles every RIFE model; only v4.6 is kept
    curl -fsSL --retry 3 -o "$TMP_ZIP" \
      "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip"
    unzip -o -q "$TMP_ZIP" -d "$TMP_DIR"
    cp "$TMP_DIR"/rife-ncnn-vulkan-*/rife-ncnn-vulkan "$RIFE_DIR/"
    cp -r "$TMP_DIR"/rife-ncnn-vulkan-*/rife-v4.6 "$RIFE_DIR/"
    cp "$TMP_DIR"/rife-ncnn-vulkan-*/LICENSE "$RIFE_DIR/" 2>/dev/null || true
    chmod +x "$RIFE_DIR/rife-ncnn-vulkan"
    rm -rf "$TMP_ZIP" "$TMP_DIR"
    echo "installed to $RIFE_DIR"
  fi
else
  echo "== Vulkan engines =="
  echo "skipping the ncnn Real-ESRGAN/RIFE binaries: no usable Vulkan GPU."
  echo "(the ONNX engine is the fast path anyway; force with --with-vulkan)"
fi

echo "== Checking engines =="
if ! .venv/bin/python - <<'PY'
import sys

from backend.engines import INTERPOLATORS, REGISTRY

available = []
for item in [*REGISTRY.values(), *INTERPOLATORS.values()]:
    reason = item.unavailable_reason()
    print(("  OK  " if reason is None else "  --  ") + item.label +
          ("" if reason is None else f" ({reason})"))
    if reason is None:
        available.append(item)
if not available:
    print("no engine or interpolator is available", file=sys.stderr)
    sys.exit(1)
PY
then
  echo "setup finished, but no engine is usable — see the reasons above." >&2
  exit 1
fi

if $HAS_NVIDIA; then
  echo "== CUDA smoke test (x2 model on the GPU) =="
  if ! .venv/bin/python - <<'PY'
import numpy as np
import onnxruntime as ort

from backend.paths import MODELS_DIR

if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit(
        "onnxruntime is installed but has no CUDAExecutionProvider: "
        + ", ".join(ort.get_available_providers())
    )
session = ort.InferenceSession(
    str(MODELS_DIR / "realesr-animevideov3-x2.onnx"),
    providers=[("CUDAExecutionProvider", {"device_id": 0})],
)
active = session.get_providers()
if not active or active[0] != "CUDAExecutionProvider":
    raise SystemExit(
        f"CUDAExecutionProvider failed to load (onnxruntime {ort.__version__}); "
        f"session fell back to {active[0] if active else 'no provider'}. "
        "The installed onnxruntime-gpu does not match this image's CUDA/cuDNN."
    )
frame = np.zeros((1, 64, 64, 3), dtype=np.uint8)
out = session.run(None, {session.get_inputs()[0].name: frame})[0]
if out.shape != (1, 128, 128, 3):
    raise SystemExit(f"unexpected output shape {out.shape}")
print(f"CUDA upscale OK: {frame.shape} -> {out.shape} via {active[0]}")
PY
  then
    echo
    echo "The CUDA smoke test failed (CUDA ${CUDA_VERSION:-unknown}," \
         "cuDNN ${CUDNN_VERSION:-unknown})." >&2
    echo "onnxruntime-gpu must match the image's CUDA/cuDNN versions." >&2
    echo "Check that 'python3' is the image's interpreter (the one with torch)," >&2
    echo "then re-run ./scripts/setup.sh." >&2
    exit 1
  fi
fi

echo
echo "Setup done."
command -v vulkaninfo >/dev/null 2>&1 && vulkan_summary \
  | grep -i "deviceName" || true
echo "Start the server with:"
echo "  ./scripts/run.sh"
echo "This binds 0.0.0.0:8000 (override with UVICORN_HOST/UVICORN_PORT)."
echo "On RunPod, expose port 8000 as an HTTP port."
