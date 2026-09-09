"""Regression tests for the exported ONNX models and the band tiling that
works around DirectML zero-filling large outputs. CPU provider only."""

import numpy as np
import pytest

from backend.engines import realesrgan_onnx as engine_mod

ort = pytest.importorskip("onnxruntime")

MODELS = [engine_mod.MODELS_DIR / f"realesr-animevideov3-x{s}.onnx" for s in (2, 3, 4)]
pytestmark = pytest.mark.skipif(not all(m.is_file() for m in MODELS), reason="ONNX models not exported")


def _session(scale):
    return ort.InferenceSession(str(engine_mod.MODELS_DIR / f"realesr-animevideov3-x{scale}.onnx"),
                                providers=["CPUExecutionProvider"])


def _gradient_frame(h, w):
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    frame = np.stack([x + 0 * y, y + 0 * x, (x + y) / 2], axis=-1).astype(np.uint8)
    return frame[None]


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_model_output_shape_and_no_black_rows(scale):
    sess = _session(scale)
    frame = _gradient_frame(48, 64)
    out = sess.run(None, {sess.get_inputs()[0].name: frame})[0]
    assert out.shape == (1, 48 * scale, 64 * scale, 3) and out.dtype == np.uint8
    assert (out[0].max(axis=(1, 2)) > 0).all(), "model produced black rows"


@pytest.mark.parametrize("scale", [2, 4])
def test_banded_inference_matches_whole_frame(scale, monkeypatch):
    sess = _session(scale)
    name = sess.get_inputs()[0].name
    frame = _gradient_frame(96, 80)
    whole = sess.run(None, {name: frame})[0]
    # budget for 56 input rows of output incl. overlap -> 24-row bands, so 96 rows = 4 bands
    monkeypatch.setattr(engine_mod, "_MAX_OUTPUT_BYTES", scale * scale * 80 * 3 * 56)
    assert engine_mod._max_band_rows(80, scale) == 24
    banded = engine_mod._infer_frame(sess, name, frame, scale)
    assert banded.shape == whole.shape
    assert np.abs(banded.astype(int) - whole.astype(int)).max() <= 1
