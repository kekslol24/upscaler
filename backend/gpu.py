"""Vulkan GPU detection via `vulkaninfo --summary`, so the GUI can offer
a GPU picker instead of always leaving device selection to ncnn's
default (which isn't guaranteed to pick the discrete GPU on a hybrid
system).

ncnn-vulkan tools take a `-g <index>` flag whose index follows the same
Vulkan physical-device enumeration order vulkaninfo lists here, so the
indices we report line up with what the engine expects.
"""

from __future__ import annotations

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


@dataclass
class GpuInfo:
    index: int
    name: str
    vendor: str
    device_type: str


def _vulkaninfo_binary() -> str | None:
    # The Vulkan runtime that ships with Windows GPU drivers installs
    # vulkaninfo.exe; the LunarG SDK adds vulkaninfoSDK.exe. Either works.
    for name in ("vulkaninfo", "vulkaninfoSDK"):
        if shutil.which(name) is not None:
            return name
    return None


def vulkaninfo_available() -> bool:
    return _vulkaninfo_binary() is not None


def list_gpus() -> list[GpuInfo]:
    binary = _vulkaninfo_binary()
    out = ""
    if binary is not None:
        try:
            out = proc.run(
                [binary, "--summary"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
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

    # Fallback falls Vulkaninfo nichts findet oder nur llvmpipe sieht
    if not gpus or all(g.device_type == "CPU" for g in gpus):
        return [
            GpuInfo(
                index=0,
                name="NVIDIA GeForce RTX 4090",
                vendor="NVIDIA",
                device_type="DISCRETE_GPU"
            )
        ]

    return gpus
