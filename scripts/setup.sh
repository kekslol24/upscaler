#!/usr/bin/env bash
# Installs system deps, sets up the Python venv, and downloads the
# realesrgan-ncnn-vulkan engine binary + models.
# Supports Arch (pacman) and Ubuntu/Debian incl. WSL2 (apt).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IS_WSL=false
if grep -qi microsoft /proc/version 2>/dev/null; then
  IS_WSL=true
fi

echo "== System packages (ffmpeg + Vulkan driver + vulkan-tools for GPU detection) =="
if command -v pacman >/dev/null; then
  sudo pacman -S --needed ffmpeg vulkan-icd-loader vulkan-tools vulkan-radeon unzip
elif command -v apt-get >/dev/null; then
  # Note: on WSL2 Ubuntu's Mesa does not ship the Dozen (Vulkan-on-D3D12)
  # driver, so Vulkan only sees the llvmpipe CPU fallback there. For GPU
  # acceleration on a Windows machine run the project natively on Windows
  # with scripts/setup.ps1 instead.
  sudo apt-get update
  sudo apt-get install -y ffmpeg libvulkan1 mesa-vulkan-drivers vulkan-tools \
    python3-venv unzip curl
else
  echo "no pacman/apt found — install ffmpeg, vulkan-tools, unzip, and your GPU's Vulkan driver manually, then re-run this script."
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
  python3 -m venv .venv
fi
.venv/bin/python -m ensurepip --upgrade >/dev/null 2>&1 || true
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo "== Real-ESRGAN ncnn-vulkan engine binary =="
ENGINE_DIR="$ROOT_DIR/engines_bin/realesrgan"
if [ -x "$ENGINE_DIR/realesrgan-ncnn-vulkan" ]; then
  echo "already installed at $ENGINE_DIR"
else
  mkdir -p "$ENGINE_DIR"
  TMP_ZIP="$(mktemp)"
  curl -sL -o "$TMP_ZIP" \
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
  curl -sL -o "$TMP_ZIP" \
    "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip"
  unzip -o -q "$TMP_ZIP" -d "$TMP_DIR"
  cp "$TMP_DIR"/rife-ncnn-vulkan-*/rife-ncnn-vulkan "$RIFE_DIR/"
  cp -r "$TMP_DIR"/rife-ncnn-vulkan-*/rife-v4.6 "$RIFE_DIR/"
  cp "$TMP_DIR"/rife-ncnn-vulkan-*/LICENSE "$RIFE_DIR/" 2>/dev/null || true
  chmod +x "$RIFE_DIR/rife-ncnn-vulkan"
  rm -rf "$TMP_ZIP" "$TMP_DIR"
  echo "installed to $RIFE_DIR"
fi

echo
echo "== Checking Vulkan sees a GPU =="
if command -v vulkaninfo >/dev/null; then
  vulkaninfo --summary 2>&1 | grep -i "deviceName" || echo "vulkaninfo ran but reported no device — check your Vulkan/Mesa install."
  if $IS_WSL; then
    echo "(on WSL the GPU shows up via the Dozen/D3D12 driver, often named"
    echo " 'Microsoft Direct3D12 (<your GPU>)'; 'llvmpipe' is the CPU fallback —"
    echo " if you only see llvmpipe, update your Windows GPU driver and run 'wsl --update')"
  fi
else
  echo "vulkaninfo not installed (optional) — install the 'vulkan-tools' package to double check GPU detection."
fi

echo
echo "Setup done. Start the server with:"
echo "  source .venv/bin/activate"
echo "  uvicorn backend.main:app --reload"
echo "Then open http://127.0.0.1:8000"
