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
  # On WSL2, mesa-vulkan-drivers provides the Dozen driver, which maps
  # Vulkan onto D3D12 and reaches the Windows GPU through /usr/lib/wsl.
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
# A venv created while python3-venv/ensurepip was missing ends up without
# its own pip. The bare `pip` then falls through to /usr/bin/pip, which on
# Debian/Ubuntu refuses with "externally-managed-environment". Recreate the
# venv in that case and always call pip through the venv's interpreter.
if [ -d .venv ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  echo "existing .venv has no working pip — recreating it"
  rm -rf .venv
fi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

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
