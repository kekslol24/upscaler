#!/usr/bin/env bash
# Starts the upscaler server, listening on all interfaces so it also works
# behind a container/proxy (RunPod, Docker). Override with UVICORN_HOST and
# UVICORN_PORT (or PORT).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -x .venv/bin/python ]; then
  echo "no .venv found — run ./scripts/setup.sh first" >&2
  exit 1
fi

# Persistent volume on RunPod; keeps uploads and renders across pod restarts.
if [ -z "${UPSCALER_DATA_DIR:-}" ] && [ -d /workspace ]; then
  export UPSCALER_DATA_DIR=/workspace/upscaler-data
fi

HOST="${UVICORN_HOST:-0.0.0.0}"
PORT="${UVICORN_PORT:-${PORT:-8000}}"

# ROCm on WSL2 needs DXG detection to enumerate the AMD GPU.
if grep -qi microsoft /proc/version 2>/dev/null; then
  export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
fi

# ROCm wheels from AMD's pip index keep their runtime in site-packages.
# PyTorch finds those through its own rpath; ONNX Runtime's ROCm provider
# needs them on the loader path too. Recent ROCm builds also need a
# librocm_smi64.so.1 alias for the librocm_smi64.so.7 shipped by the
# system/PyTorch, so point a cache-dir symlink at whichever .7 exists.
ROCM_SHIM_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/upscaler/rocm-libs"
for rocm_lib in \
  .venv/lib/python*/site-packages/_rocm_sdk_core/lib \
  .venv/lib/python*/site-packages/_rocm_sdk_libraries/lib; do
  [ -d "$rocm_lib" ] && export LD_LIBRARY_PATH="$rocm_lib:${LD_LIBRARY_PATH:-}"
done
for rocm_smi in /opt/rocm*/lib/librocm_smi64.so.7 \
  .venv/lib/python*/site-packages/torch/lib/librocm_smi64.so.7; do
  [ -e "$rocm_smi" ] || continue
  mkdir -p "$ROCM_SHIM_DIR"
  ln -sf "$rocm_smi" "$ROCM_SHIM_DIR/librocm_smi64.so.1"
  export LD_LIBRARY_PATH="$ROCM_SHIM_DIR:${LD_LIBRARY_PATH:-}"
  break
done

echo "Upscaler listening on http://${HOST}:${PORT}"
[ -n "${UPSCALER_DATA_DIR:-}" ] && echo "Job data: ${UPSCALER_DATA_DIR}"

# The NVIDIA Vulkan ICD can fail with vkCreateInstance -9 when no X display
# exists, even though the container is headless. RIFE (and the Vulkan
# upscalers) need a working Vulkan device, so give them a virtual one.
if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null 2>&1; then
  echo "Starting a virtual X display for headless Vulkan"
  exec xvfb-run -a .venv/bin/python -m uvicorn backend.main:app --host "$HOST" --port "$PORT"
fi
exec .venv/bin/python -m uvicorn backend.main:app --host "$HOST" --port "$PORT"
