# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the one-folder desktop build.

    pyinstaller upscaler.spec

One-folder (not one-file) on purpose: a one-file exe unpacks a couple of
hundred MB into a temp directory on *every* launch, which is where most of
the slow-start complaints about packaged Python apps come from. Here the
runtime sits in dist/Upscaler/_internal and starts in about a second.

Heavy, updatable assets (ffmpeg, engine binaries, ONNX models) are NOT
bundled by this spec — scripts/build_exe.ps1 copies them next to the exe,
where backend/paths.py looks for them.
"""

import sys

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

hiddenimports = (
    collect_submodules("uvicorn")
    + ["backend", "backend.main", "backend.jobs", "backend.pipeline", "backend.gpu",
       "backend.presets", "backend.paths", "backend.proc", "backend.engines",
       "backend.engines.realesrgan_onnx", "backend.engines.realesrgan_ncnn",
       "backend.engines.rife_ncnn",
       "multipart", "python_multipart", "numpy"]
)

binaries = []
try:
    import onnxruntime  # noqa: F401
    # DirectML.dll and onnxruntime's own DLLs live inside the package
    binaries += collect_dynamic_libs("onnxruntime")
    hiddenimports += ["onnxruntime", "onnxruntime.capi", "onnxruntime.capi._pybind_state"]
except ImportError:
    pass

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=[("frontend", "frontend")],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PIL", "torch", "torchvision", "onnx", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="Upscaler",
    debug=False,
    strip=False,
    upx=False,
    console=sys.platform != "win32",  # log file + message box on Windows, terminal elsewhere
    icon="installer/upscaler.ico" if sys.platform == "win32" else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Upscaler",
)
