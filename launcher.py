"""Desktop entry point for the packaged build: start the server on a
free local port, open the browser, keep running until the window/process
is closed. Errors go to a log file in the data directory (there is no
console in the packaged build) and to a message box on Windows.

Also runnable from a checkout:  python launcher.py
"""

from __future__ import annotations

import logging
import socket
import sys
import threading
import webbrowser
from pathlib import Path


def _free_port(preferred: int = 8000) -> int:
    for port in (preferred, *range(preferred + 1, preferred + 20)):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 0


def _open_when_ready(url: str, port: int, timeout: float = 60.0) -> None:
    """Open the browser once the server accepts connections, so the first
    page load doesn't race the (slower, packaged) startup."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            try:
                s.connect(("127.0.0.1", port))
                break
            except OSError:
                time.sleep(0.25)
    webbrowser.open(url)


def _fatal(message: str) -> None:
    logging.exception(message)
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, f"{message}\n\nSee upscaler.log in the data folder.", "Upscaler", 0x10)
    sys.exit(1)


def main() -> None:
    from backend import paths  # sets up PATH for the bundled ffmpeg

    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_file = paths.DATA_DIR / "upscaler.log"
    handlers: list[logging.Handler] = [logging.FileHandler(log_file, encoding="utf-8")]
    if sys.stderr is not None:  # absent in a windowed (console=False) build
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    logging.info("starting; data dir %s, app dir %s, frozen=%s", paths.DATA_DIR, paths.APP_DIR, paths.FROZEN)

    try:
        import uvicorn
        from backend.main import app
    except Exception:  # noqa: BLE001
        _fatal("Failed to load the application")
        return

    port = _free_port()
    if not port:
        _fatal("No free local port between 8000 and 8020")
        return
    url = f"http://127.0.0.1:{port}"
    if "--no-browser" not in sys.argv:
        threading.Thread(target=_open_when_ready, args=(url, port), daemon=True).start()
    logging.info("serving on %s", url)
    print(f"Upscaler running at {url} — close this window to stop.", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info", log_config=None)


if __name__ == "__main__":
    if not getattr(sys, "frozen", False):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
