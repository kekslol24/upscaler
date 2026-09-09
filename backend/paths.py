"""Where things live, both when run from a checkout and when frozen into
a PyInstaller one-folder build (see upscaler.spec / scripts/build_exe.ps1).

Frozen layout, all next to the executable:

    Upscaler.exe
    _internal/          python runtime + packages + bundled frontend/
    bin/                ffmpeg.exe, ffprobe.exe
    engines_bin/        ncnn binaries + their models (realesrgan/, rife/)
    models/             ONNX models

Job data goes to a per-user writable directory, never next to the exe.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))

if FROZEN:
    APP_DIR = Path(sys.executable).resolve().parent
    #: PyInstaller extracts bundled data files (frontend/) here
    RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
else:
    APP_DIR = Path(__file__).resolve().parent.parent
    RESOURCE_DIR = APP_DIR

FRONTEND_DIR = RESOURCE_DIR / "frontend"
ENGINES_DIR = APP_DIR / "engines_bin"
MODELS_DIR = APP_DIR / "models"


def _default_data_dir() -> Path:
    override = os.environ.get("UPSCALER_DATA_DIR")
    if override:
        return Path(override)
    if not FROZEN:
        return APP_DIR / "data"
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "Upscaler"


DATA_DIR = _default_data_dir()
JOBS_DIR = DATA_DIR / "jobs"


def add_bundled_tools_to_path() -> None:
    """Make a bundled ffmpeg (APP_DIR/bin) win over whatever is on PATH."""
    tools = APP_DIR / "bin"
    if tools.is_dir():
        os.environ["PATH"] = str(tools) + os.pathsep + os.environ.get("PATH", "")


add_bundled_tools_to_path()
