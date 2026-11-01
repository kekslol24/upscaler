"""GPU detection for the GPU picker.

Two sources are consulted:

- `vulkaninfo --summary`: the ncnn-vulkan binaries take a `-g <index>`
  flag whose index follows Vulkan's physical-device enumeration order,
  so those indices are what the Vulkan engines expect.
- `nvidia-smi`: the ONNX Runtime CUDA provider numbers devices in CUDA
  order (honouring CUDA_VISIBLE_DEVICES), which is what `device_id`
  expects. This is also the only detection available on headless Linux
  containers (RunPod and friends), where there is no Vulkan driver.
- `rocminfo`: AMD's ROCm user-space tools. ROCm PyTorch exposes the same
  `torch.cuda` API, so its device index is what the RIFE/ONNX AMD paths
  expect. This covers native Linux and WSL2 installs where Vulkan is not
  installed or only exposes a CPU/llvmpipe device.

A real Vulkan GPU wins when there is one, because that list is already
what the Vulkan engines use. Otherwise NVIDIA is used, then AMD/ROCm, so
the CUDA/ROCm engines can be selected on machines without Vulkan.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass

from . import proc

_VENDOR_IDS = {
    "0x1002": "AMD",
    "0x10de": "NVIDIA",
    "0x8086": "Intel",
}

_GPU_HEADER_RE = re.compile(r"^GPU(\d+):")
_FIELD_RE = re.compile(r"^\s*(\w+)\s*=\s*(.+)$")
_AGENT_RE = re.compile(r"^Agent\s+\d+\s*$")
# rocminfo's top-level agent fields are indented by exactly two spaces;
# nested ISA/cache fields are indented further and must not be picked up.
_ROCM_FIELD_RE = re.compile(r"^  ([A-Za-z][\w ]*?):\s+(.+?)\s*$")


@dataclass
class GpuInfo:
    index: int
    name: str
    vendor: str
    device_type: str


@dataclass
class _NvidiaGpu:
    index: int
    name: str
    uuid: str


def _vulkaninfo_binary() -> str | None:
    # The Vulkan runtime that ships with Windows GPU drivers installs
    # vulkaninfo.exe; the LunarG SDK adds vulkaninfoSDK.exe. Either works.
    for name in ("vulkaninfo", "vulkaninfoSDK"):
        if shutil.which(name) is not None:
            return name
    return None


def _nvidia_smi_binary() -> str | None:
    return shutil.which("nvidia-smi")


def _rocminfo_binary() -> str | None:
    return shutil.which("rocminfo")


def rocm_available() -> bool:
    """True if AMD's ROCm user-space tools are installed.

    This does not mean a usable GPU is present — for that see `list_gpus()`.
    The RIFE diagnostics use it to tell an AMD machine with a missing
    PyTorch/ROCm build from an NVIDIA machine with a missing CUDA driver.
    """
    return any(
        shutil.which(name) is not None
        for name in ("rocminfo", "rocm-smi", "amd-smi")
    )


def vulkaninfo_available() -> bool:
    return _vulkaninfo_binary() is not None


def detection_available() -> bool:
    """True if any backend tool is installed, so the picker can be
    populated. An empty list with detection on means "no GPU found"."""
    return (
        vulkaninfo_available()
        or _nvidia_smi_binary() is not None
        or rocm_available()
    )


def cuda_visible_devices() -> str | None:
    """Raw value of CUDA_VISIBLE_DEVICES, the env var that decides which
    physical GPUs PyTorch/ONNX see as `cuda:0..N`. `None` means unset."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    return raw.strip() or "(empty)"


def _list_vulkan_gpus() -> list[GpuInfo]:
    binary = _vulkaninfo_binary()
    out = ""
    if binary is not None:
        try:
            out = proc.run(
                [binary, "--summary"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            out = ""

    gpus: list[GpuInfo] = []
    current_index: int | None = None
    fields: dict[str, str] = {}

    def flush() -> None:
        if current_index is not None and "deviceName" in fields:
            vendor_id = fields.get("vendorID", "").split()[0].lower()
            gpus.append(GpuInfo(
                index=current_index,
                name=fields["deviceName"],
                vendor=_VENDOR_IDS.get(vendor_id, vendor_id or "unknown"),
                device_type=fields.get("deviceType", "").removeprefix("PHYSICAL_DEVICE_TYPE_"),
            ))

    for line in out.splitlines():
        header = _GPU_HEADER_RE.match(line.strip())
        if header:
            flush()
            current_index = int(header.group(1))
            fields = {}
            continue
        field = _FIELD_RE.match(line)
        if field and current_index is not None:
            fields[field.group(1)] = field.group(2).strip()
    flush()
    return gpus


def _nvidia_gpus() -> list[_NvidiaGpu]:
    binary = _nvidia_smi_binary()
    if binary is None:
        return []
    try:
        out = proc.run(
            [binary, "--query-gpu=index,name,uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []

    gpus: list[_NvidiaGpu] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        gpus.append(_NvidiaGpu(int(parts[0]), parts[1], parts[2]))
    return gpus


def _visible_nvidia_gpus(gpus: list[_NvidiaGpu]) -> list[_NvidiaGpu]:
    """Apply CUDA_VISIBLE_DEVICES so the reported index matches the
    logical CUDA device number onnxruntime's device_id expects."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return gpus
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts or parts == ["-1"]:
        return []
    by_index = {str(g.index): g for g in gpus}
    by_uuid = {g.uuid.lower(): g for g in gpus}
    selected: list[_NvidiaGpu] = []
    for part in parts:
        match = by_index.get(part) or by_uuid.get(part.lower())
        if match is None and not part.isdigit():
            # CUDA also accepts abbreviated UUIDs (GPU-xxx)
            match = next(
                (g for g in gpus if g.uuid.lower().startswith(part.lower())),
                None,
            )
        if match is not None:
            selected.append(match)
    return selected


def _list_nvidia_gpus() -> list[GpuInfo]:
    gpus = _visible_nvidia_gpus(_nvidia_gpus())
    return [
        GpuInfo(
            index=index,
            name=gpu.name,
            vendor="NVIDIA",
            device_type="DISCRETE_GPU",
        )
        for index, gpu in enumerate(gpus)
    ]


def _rocm_gpu_agents(out: str) -> list[GpuInfo]:
    """Parse the GPU agents out of `rocminfo` output."""
    gpus: list[GpuInfo] = []
    name = marketing = device_type = None

    def flush() -> None:
        nonlocal name, marketing, device_type
        if device_type == "GPU" and (marketing or name):
            gpus.append(GpuInfo(
                index=len(gpus),
                name=marketing or name,
                vendor="AMD",
                # rocminfo does not distinguish discrete/integrated.
                device_type="DISCRETE_GPU",
            ))
        name = marketing = device_type = None

    for line in out.splitlines():
        if _AGENT_RE.match(line.strip()):
            flush()
            continue
        field = _ROCM_FIELD_RE.match(line)
        if field is None:
            continue
        key, value = field.group(1), field.group(2)
        if key == "Name":
            name = value
        elif key == "Marketing Name":
            marketing = value
        elif key == "Device Type":
            device_type = value
    flush()
    return gpus


def _list_rocm_gpus() -> list[GpuInfo]:
    binary = _rocminfo_binary()
    if binary is None:
        return []
    env = dict(os.environ)
    # WSL2 exposes the GPU through /dev/dxg; ROCm only enumerates it with
    # DXG detection switched on (the ROCm torch wheels set this too).
    env.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
    try:
        out = proc.run(
            [binary],
            capture_output=True, text=True, timeout=15, check=True, env=env,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []
    return _rocm_gpu_agents(out)


def list_gpus() -> list[GpuInfo]:
    vulkan = _list_vulkan_gpus()
    hardware = [g for g in vulkan if g.device_type in ("DISCRETE_GPU", "INTEGRATED_GPU")]
    if hardware:
        # A headless Xvfb setup also makes Mesa's llvmpipe visible; it is a
        # CPU device, not something the GPU picker should offer.
        return hardware

    nvidia = _list_nvidia_gpus()
    if nvidia:
        return nvidia

    rocm = _list_rocm_gpus()
    if rocm:
        return rocm

    # No hardware backend: expose whatever Vulkan reported (possibly the
    # llvmpipe CPU device) rather than inventing a GPU that isn't there.
    return vulkan
