"""Helpers for reaching a Windows ffmpeg from WSL2.

The Linux ffmpeg under WSL has no AMF/NVENC; the Windows build can encode
with the native AMD/NVIDIA driver. Paths must be translated to Windows
form, and ffmpeg launched through the WSL interop layer can still read a
raw-frame pipe from the Linux side.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import proc


def is_wsl() -> bool:
    if sys.platform == "win32":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text(errors="ignore").lower()
    except OSError:
        return False


@functools.lru_cache(maxsize=1)
def windows_ffmpeg() -> str | None:
    if not is_wsl():
        return None
    cmd = shutil.which("cmd.exe") or "/mnt/c/Windows/System32/cmd.exe"
    if not os.path.isfile(cmd):
        return None
    try:
        out = proc.run(
            [cmd, "/c", "where", "ffmpeg"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, errors="replace", timeout=20,
        ).stdout
    except (subprocess.TimeoutExpired, OSError):
        return None
    for line in out.splitlines():
        win_path = line.strip()
        if not win_path:
            continue
        try:
            host_path = proc.run(
                ["wslpath", "-u", win_path],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            continue
        if host_path and os.path.isfile(host_path):
            return host_path
    return None


def host_path(path: Path) -> str:
    """Translate a path for a Windows ffmpeg (e.g. to a UNC path)."""
    if not is_wsl():
        return str(path)
    try:
        return proc.run(
            ["wslpath", "-w", str(path)],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return str(path)


@functools.lru_cache(maxsize=8)
def encoder_available(binary: str, name: str) -> bool:
    try:
        out = proc.run(
            [binary, "-hide_banner", "-encoders"],
            capture_output=True, text=True, errors="replace",
            timeout=20, check=True,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return f" {name} " in out
