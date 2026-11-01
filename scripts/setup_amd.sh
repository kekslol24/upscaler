#!/usr/bin/env bash
# AMD (Radeon) one-shot setup for the 4K Upscaler.
#
#   ./scripts/setup_amd.sh              # set up Vulkan (+ ROCm if already usable), then start
#   ./scripts/setup_amd.sh --rocm       # also install ROCm PyTorch for gfx1200
#   ./scripts/setup_amd.sh --vulkan     # Vulkan only, never touch ROCm
#   ./scripts/setup_amd.sh --no-start   # set up only, don't start the server
#   ./scripts/setup_amd.sh --yes        # non-interactive (accept apt/pip prompts)
#
# Works on native Linux and WSL2 (Ubuntu 22.04/24.04). On WSL2:
#   - Vulkan needs Mesa's Dozen driver (installed here via the kisak-mesa PPA).
#   - The fast PyTorch RIFE path needs ROCm + ROCDXG (librocdxg) and the AMD
#     Adrenalin WSL2 driver on Windows. Use --rocm once that is in place.
#
# Everything is idempotent: re-running it is safe.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE=auto            # auto | vulkan | rocm
START=1
ASSUME_YES=0
ROCM_INDEX="${ROCM_INDEX:-https://stable.repo.amd.com/rocm/whl-next/}"
ROCM_TORCH="${ROCM_TORCH:-torch[device-gfx1200]==2.14.0+rocm10.1.0}"
ROCM_TORCHVISION="${ROCM_TORCHVISION:-torchvision[device-gfx1200]==0.29.0a0+rocm10.1.0}"
ROCM_TORCHAUDIO="${ROCM_TORCHAUDIO:-torchaudio==2.11.0.3+rocm10.1.0}"

usage() {
    cat <<'EOF'
AMD (Radeon) one-shot setup for the 4K Upscaler.

  ./scripts/setup_amd.sh              # set up Vulkan (+ ROCm if already usable), then start
  ./scripts/setup_amd.sh --rocm       # also install ROCm PyTorch for gfx1200
  ./scripts/setup_amd.sh --vulkan     # Vulkan only, never touch ROCm
  ./scripts/setup_amd.sh --no-start   # set up only, don't start the server
  ./scripts/setup_amd.sh --yes        # non-interactive (accept apt/pip prompts)

Works on native Linux and WSL2 (Ubuntu 22.04/24.04). On WSL2:
  - Vulkan needs Mesa's Dozen driver (installed here via the kisak-mesa PPA).
  - The fast PyTorch RIFE path needs ROCm + ROCDXG (librocdxg) and the AMD
    Adrenalin WSL2 driver on Windows. Use --rocm once that is in place.

Everything is idempotent: re-running it is safe.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --vulkan) MODE=vulkan ;;
        --rocm) MODE=rocm ;;
        --no-start) START=0 ;;
        --yes|-y) ASSUME_YES=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        echo "error: run as root or install sudo." >&2
        exit 1
    fi
fi

IS_WSL=false
if grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=true
    # WSLg ships the D3D12/DXCore libraries here; Dozen and ncnn need them.
    if [ -d /usr/lib/wsl/lib ]; then
        export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
    fi
    export HSA_ENABLE_DXG_DETECTION=1
fi

rocm_runtime_env() {
    # AMD's pip wheels keep ROCm in site-packages. ONNX Runtime loads its
    # provider with dlopen, so these paths must be in the environment before
    # Python starts. Recent builds also need librocm_smi64.so.1 while the
    # system/PyTorch ship librocm_smi64.so.7; a cache-dir symlink bridges them.
    local lib smi
    local shim_dir="${XDG_CACHE_HOME:-$HOME/.cache}/upscaler/rocm-libs"
    for lib in "$ROOT_DIR"/.venv/lib/python*/site-packages/_rocm_sdk_core/lib \
        "$ROOT_DIR"/.venv/lib/python*/site-packages/_rocm_sdk_libraries/lib; do
        [ -d "$lib" ] && export LD_LIBRARY_PATH="$lib:${LD_LIBRARY_PATH:-}"
    done
    for smi in /opt/rocm*/lib/librocm_smi64.so.7 \
        "$ROOT_DIR"/.venv/lib/python*/site-packages/torch/lib/librocm_smi64.so.7; do
        [ -e "$smi" ] || continue
        mkdir -p "$shim_dir"
        ln -sf "$smi" "$shim_dir/librocm_smi64.so.1"
        export LD_LIBRARY_PATH="$shim_dir:${LD_LIBRARY_PATH:-}"
        break
    done
}

rocm_runtime_env

confirm() {
    if [ "$ASSUME_YES" -eq 1 ]; then
        return 0
    fi
    local answer
    read -r -p "$1 [y/N] " answer
    case "$answer" in
        y|Y|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

apt_install() {
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
}

vulkan_hw() {
    command -v vulkaninfo >/dev/null 2>&1 || return 1
    local out
    out="$(vulkaninfo --summary 2>/dev/null || true)"
    printf '%s\n' "$out" \
        | grep -Eq 'PHYSICAL_DEVICE_TYPE_(DISCRETE_GPU|INTEGRATED_GPU)' || return 1
    printf '%s\n' "$out" | grep -Eqi 'amd|radeon|direct3d12'
}

ensure_vulkan() {
    if vulkan_hw; then
        echo "== Vulkan already sees an AMD GPU =="
        return 0
    fi
    if [ "$MODE" = rocm ]; then
        echo "== --rocm selected; Vulkan is optional and will be skipped =="
        return 0
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "warning: no apt-get here; install Mesa Vulkan drivers manually." >&2
        return 1
    fi

    if $IS_WSL; then
        cat <<'EOF'
== WSL2 Vulkan setup (Mesa Dozen / Vulkan-on-D3D12) ==
Ubuntu's stock Mesa does not include the dzn driver, so Vulkan only sees
llvmpipe (CPU). This adds the kisak-mesa PPA and upgrades mesa-vulkan-drivers.
EOF
        confirm "Install Mesa Dozen now?" || { echo "skipped Vulkan setup"; return 1; }
        apt_install software-properties-common
        $SUDO add-apt-repository -y ppa:kisak/kisak-mesa
        $SUDO apt-get update
        apt_install mesa-vulkan-drivers vulkan-tools
        $SUDO apt-get install -y --only-upgrade mesa-vulkan-drivers || true
    else
        confirm "Install Mesa Vulkan drivers (mesa-vulkan-drivers, vulkan-tools)?" \
            || { echo "skipped Vulkan setup"; return 1; }
        apt_install libvulkan1 mesa-vulkan-drivers vulkan-tools
    fi

    if vulkan_hw; then
        echo "== Vulkan GPU detected =="
        return 0
    fi
    echo "warning: Vulkan still does not see the AMD GPU." >&2
    echo "         On WSL2 try: sudo apt upgrade, then re-run this script." >&2
    return 1
}

torch_ok() {
    [ -x .venv/bin/python ] || return 1
    .venv/bin/python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

rocm_ready() {
    if $IS_WSL; then
        [ -e /dev/dxg ] || return 1
        # Current ROCm talks to the WSL GPU paravirt device directly, but
        # only enumerates it with DXG detection enabled. Older setups also
        # need ROCDXG; accept either.
        if command -v rocminfo >/dev/null 2>&1 \
            && HSA_ENABLE_DXG_DETECTION=1 rocminfo 2>/dev/null \
                | grep -qE 'Device Type:[[:space:]]+GPU'; then
            return 0
        fi
        if ldconfig -p 2>/dev/null | grep -qi librocdxg; then
            return 0
        fi
        if [ -e /opt/rocm/lib/librocdxg.so ]; then
            return 0
        fi
        return 1
    fi
    [ -e /dev/kfd ] || return 1
    return 0
}

rocm_hint() {
    if $IS_WSL; then
        cat >&2 <<'EOF'
WSL2 ROCm needs two things outside this repo:
  1. AMD Adrenalin WSL2 driver on Windows (26.x or newer).
  2. ROCDXG (librocdxg) inside WSL2:
       https://github.com/ROCm/librocdxg
Then re-run:  ./scripts/setup_amd.sh --rocm
EOF
    else
        cat >&2 <<'EOF'
ROCm needs:
  - ROCm 10.1 installed for gfx1200 (RX 9060 XT), and
  - your user in the render/video groups (log out/in after adding).
Then re-run:  ./scripts/setup_amd.sh --rocm
EOF
    fi
}

ensure_rocm() {
    if torch_ok; then
        echo "== PyTorch GPU already available =="
        .venv/bin/python -c \
            'import torch; print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))'
        return 0
    fi
    if [ "$MODE" = vulkan ]; then
        echo "== --vulkan selected; skipping ROCm =="
        return 0
    fi
    if [ "$MODE" = auto ] && ! rocm_ready; then
        echo "== ROCm/ROCDXG not detected; staying on Vulkan =="
        echo "   The app runs, but the fast PyTorch RIFE path needs ROCm."
        rocm_hint
        return 0
    fi
    if ! rocm_ready; then
        echo "error: --rocm was requested but ROCm/ROCDXG is not available." >&2
        rocm_hint
        return 1
    fi

    confirm "Install ROCm PyTorch for gfx1200 into .venv? This downloads several GB." \
        || { echo "skipped ROCm install"; return 1; }
    .venv/bin/python -m pip install --index-url "$ROCM_INDEX" \
        "$ROCM_TORCH" "$ROCM_TORCHVISION" "$ROCM_TORCHAUDIO"

    if torch_ok; then
        echo "== ROCm PyTorch installed =="
        return 0
    fi
    echo "error: torch installed but CUDA/ROCm is not usable." >&2
    rocm_hint
    return 1
}

torch_hip_major() {
    [ -x .venv/bin/python ] || return 1
    .venv/bin/python - <<'PY' 2>/dev/null || true
import torch
hip = getattr(torch.version, "hip", None) or ""
print(hip.split(".")[0] if hip else "")
PY
}

ort_gpu_ok() {
    [ -x .venv/bin/python ] || return 1
    .venv/bin/python - <<'PY' >/dev/null 2>&1
import onnxruntime as ort
gpu = {"MIGraphXExecutionProvider", "ROCMExecutionProvider"}
raise SystemExit(0 if gpu.intersection(ort.get_available_providers()) else 1)
PY
}

ensure_ort() {
    if ! rocm_ready; then
        echo "== ROCm not available; skipping ONNX Runtime ROCm provider =="
        return 0
    fi
    # The torch install above may have added site-packages/ROCm since the
    # top-level call, so refresh the loader path before running Python.
    rocm_runtime_env

    # ORT's AMD Linux wheels must match the HIP runtime. The gfx1200 PyTorch
    # wheel ships ROCm 10.1 (libamdhip64.so.7), so use the ROCm 7 build;
    # native ROCm 6 installs need the older post1 wheel (libamdhip64.so.6).
    local major spec
    major="$(torch_hip_major)"
    if [ -z "$major" ] && [ -r /opt/rocm/.info/version ]; then
        major="$(cut -d. -f1 /opt/rocm/.info/version)"
    fi
    case "$major" in
        ''|*[!0-9]*) major=0 ;;
    esac
    if [ "$major" -ge 7 ]; then
        spec="onnxruntime-rocm==1.22.2.post3"
    else
        spec="onnxruntime-rocm==1.22.2.post1"
    fi

    if ort_gpu_ok && .venv/bin/python -m pip show onnxruntime-rocm 2>/dev/null \
        | grep -q "^Version: ${spec##*==}$"; then
        echo "== ONNX Runtime ROCm provider already available =="
        return 0
    fi

    echo "== ONNX Runtime AMD provider ($spec) =="
    # CPU onnxruntime and onnxruntime-gpu shadow the ROCm module at import.
    for pkg in onnxruntime onnxruntime-gpu onnxruntime-rocm \
        onnxruntime-migraphx onnxruntime-ep-migraphx; do
        .venv/bin/python -m pip uninstall -y "$pkg" >/dev/null 2>&1 || true
    done
    .venv/bin/python -m pip install "$spec" || {
        echo "warning: could not install $spec; the ONNX upscaler will be" >&2
        echo "         unavailable and the Vulkan upscaler remains usable." >&2
        return 1
    }
    .venv/bin/python - <<'PY' || true
import onnxruntime as ort
from pathlib import Path
providers = [p for p in ("MIGraphXExecutionProvider", "ROCMExecutionProvider")
             if p in ort.get_available_providers()]
if not providers:
    raise SystemExit(1)
model = Path("models/realesr-animevideov3-x2.onnx")
if model.exists():
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(model), opts, providers=providers)
    print("  onnxruntime", ort.__version__, sess.get_providers())
else:
    print("  onnxruntime", ort.__version__, providers)
PY
}

# -- detect the machine ------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "NVIDIA GPU detected — this script is for AMD."
    echo "Use ./scripts/setup.sh instead."
    exit 0
fi

AMD_NAME=""
if command -v lspci >/dev/null 2>&1; then
    AMD_NAME="$(lspci 2>/dev/null | grep -Ei 'vga|3d|display' \
        | grep -Ei 'amd|ati|radeon' | head -n1 || true)"
fi
if [ -z "$AMD_NAME" ] && command -v vulkaninfo >/dev/null 2>&1; then
    AMD_NAME="$(vulkaninfo --summary 2>/dev/null \
        | grep -Ei 'deviceName.*(amd|radeon|direct3d12)' | head -n1 || true)"
fi
if [ -n "$AMD_NAME" ]; then
    echo "== AMD GPU: $AMD_NAME =="
else
    echo "warning: no AMD GPU found via lspci/vulkaninfo; continuing anyway." >&2
fi
if $IS_WSL; then
    echo "== WSL2 detected =="
fi

# -- Vulkan, then the normal Linux setup ------------------------------------
ensure_vulkan || true

echo
echo "== Base setup (ffmpeg, venv, deps, RIFE weights, ncnn engines) =="
set +e
./scripts/setup.sh --with-vulkan
SETUP_RC=$?
set -e
if [ "$SETUP_RC" -ne 0 ]; then
    echo "warning: scripts/setup.sh exited $SETUP_RC (often 'no engine usable yet');" >&2
    echo "         continuing so ROCm can be installed." >&2
fi

# -- optional ROCm -----------------------------------------------------------
echo
echo "== ROCm / PyTorch =="
ensure_rocm || true

# -- ONNX Runtime for AMD ----------------------------------------------------
echo
echo "== ONNX Runtime =="
ensure_ort || true

# -- report what is usable ---------------------------------------------------
echo
echo "== Engine availability =="
if [ -x .venv/bin/python ]; then
    set +e
    .venv/bin/python - <<'PY'
from backend.engines import INTERPOLATORS, REGISTRY

ok = False
for item in [*REGISTRY.values(), *INTERPOLATORS.values()]:
    reason = item.unavailable_reason()
    print(("  OK  " if reason is None else "  --  ") + item.label
          + ("" if reason is None else f" ({reason})"))
    ok = ok or reason is None
raise SystemExit(0 if ok else 1)
PY
    ENGINES_RC=$?
    set -e
else
    ENGINES_RC=1
fi
if [ "$ENGINES_RC" -ne 0 ]; then
    echo "error: no engine is usable. See the reasons above." >&2
    exit 1
fi

# -- start -------------------------------------------------------------------
if [ "$START" -eq 1 ]; then
    echo
    echo "Starting the server — open http://localhost:8000"
    exec ./scripts/run.sh
else
    echo
    echo "Setup done. Start with: ./scripts/run.sh"
fi
