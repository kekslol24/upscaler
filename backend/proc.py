"""subprocess.run / Popen that never open a console window.

The packaged Windows build is a windowed app with no console of its own,
so every ffmpeg/ffprobe/ncnn child it starts would otherwise pop up a
black cmd window. CREATE_NO_WINDOW suppresses that; on other platforms
these are plain pass-throughs.
"""

from __future__ import annotations

import subprocess
import sys

_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def run(args, **kwargs) -> subprocess.CompletedProcess:
    # a child with no usable stdin handle can misbehave on Windows
    if "stdin" not in kwargs and "input" not in kwargs:
        kwargs["stdin"] = subprocess.DEVNULL
    return subprocess.run(args, **_FLAGS, **kwargs)


def Popen(args, **kwargs) -> subprocess.Popen:  # noqa: N802 - mirrors subprocess
    if "stdin" not in kwargs:
        kwargs["stdin"] = subprocess.DEVNULL
    return subprocess.Popen(args, **_FLAGS, **kwargs)
