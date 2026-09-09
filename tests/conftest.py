import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory) -> Path:
    """A 64x48, 12-frame test pattern with an Opus audio track — the
    awkward MKV-style source that exercises the audio transcode path."""
    path = tmp_path_factory.mktemp("src") / "tiny.mkv"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=64x48:rate=12:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "libopus",
            str(path),
        ],
        check=True,
    )
    return path
