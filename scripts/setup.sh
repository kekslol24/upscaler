#!/usr/bin/env bash
# Run this on your actual Arch Linux laptop (not inside a sandboxed dev
# session) — it installs system deps, sets up the Python venv, and
# downloads the realesrgan-ncnn-vulkan engine binary + models.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== System packages (ffmpeg + Vulkan for AMD/RADV + vulkan-tools for GPU detection) =="
if command -v pacman >/dev/null; then
  sudo pacman -S --needed ffmpeg vulkan-icd-loader vulkan-tools vulkan-radeon
else
  echo "pacman not found — install ffmpeg, vulkan-tools, and your GPU's Vulkan driver manually, then re-run this script."
fi

echo "== Python venv =="
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

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
else
  echo "vulkaninfo not installed (optional) — install the 'vulkan-tools' package to double check GPU detection."
fi

echo
echo "Setup done. Start the server with:"
echo "  source .venv/bin/activate"
echo "  uvicorn backend.main:app --reload"
echo "Then open http://127.0.0.1:8000"
