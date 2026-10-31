"""Unit tests for the GPU / CUDA diagnostics surfaced to the UI.

These check the pure, hardware-free parts: the CUDA_VISIBLE_DEVICES helper,
and the RIFE interpolator's error messages for the no-CUDA / zero-device
cases. Real detect/render still needs a GPU and lives in test_stages.py.
"""

import pytest

from backend import gpu
from backend.engines import rife_torch
from backend.engines.rife_torch import _cuda_state


def test_cuda_visible_devices_unset(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gpu.cuda_visible_devices() is None


@pytest.mark.parametrize("raw,expected", [
    ("0,1,2", "0,1,2"),
    ("", "(empty)"),
    ("   ", "(empty)"),
    ("GPU-a1b2c3", "GPU-a1b2c3"),
])
def test_cuda_visible_devices(monkeypatch, raw, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", raw)
    assert gpu.cuda_visible_devices() == expected


############ RIFE interpolator diagnostics (hardware-free) ############


@pytest.fixture
def interp(monkeypatch, tmp_path):
    i = rife_torch.RifeTorchInterpolator("rife-v4.6", "RIFE v4.6")
    fake_weights = tmp_path / "flownet.pkl"
    fake_weights.write_bytes(b"fake")
    monkeypatch.setattr(i, "model_path", fake_weights)
    return i


@pytest.mark.parametrize("state,expected_substring", [
    ({"error": None, "available": False, "device_count": 0, "visible": None}, "PyTorch CUDA is not available"),
    ({"error": None, "available": False, "device_count": 0, "visible": "1,3"}, "1,3"),
    ({"error": "PyTorch reports zero CUDA devices despite CUDA being available", "available": True, "device_count": 0, "visible": None}, "zero CUDA devices"),
])
def test_unavailable_reason_messages(monkeypatch, interp, state, expected_substring):
    monkeypatch.setattr(rife_torch, "_cuda_state", lambda: state)
    reason = interp.unavailable_reason()
    assert reason is not None
    assert expected_substring in reason


@pytest.mark.parametrize("state,expected", [
    ({"error": None, "available": True, "device_count": 2, "visible": None}, None),
    ({"error": None, "available": False, "device_count": 0, "visible": None}, "PyTorch CUDA is not available"),
])
def test_unavailable_reason_ok_and_nocuda(monkeypatch, interp, state, expected):
    monkeypatch.setattr(rife_torch, "_cuda_state", lambda: state)
    reason = interp.unavailable_reason()
    if expected is None:
        assert reason is None
    else:
        assert expected in reason


def test_is_available_reflects_cuda_state(monkeypatch, interp):
    monkeypatch.setattr(rife_torch, "_cuda_state", lambda: {"ok": True})
    assert interp.is_available() is True
    monkeypatch.setattr(rife_torch, "_cuda_state", lambda: {"ok": False})
    assert interp.is_available() is False
