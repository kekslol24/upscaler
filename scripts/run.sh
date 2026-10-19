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

echo "Upscaler listening on http://${HOST}:${PORT}"
[ -n "${UPSCALER_DATA_DIR:-}" ] && echo "Job data: ${UPSCALER_DATA_DIR}"
exec .venv/bin/python -m uvicorn backend.main:app --host "$HOST" --port "$PORT"
