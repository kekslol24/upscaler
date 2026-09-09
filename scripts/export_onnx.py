"""Export Real-ESRGAN's realesr-animevideov3 weights to ONNX for the
DirectML engine (backend/engines/realesrgan_onnx.py).

The graph takes a uint8 RGB frame (1, H, W, 3) and returns a uint8 RGB
frame, so the runtime can stream raw frames straight between ffmpeg and
the model without any float conversion in Python.

The ncnn x2/x3 variants shipped with realesrgan-ncnn-vulkan are the x4
network followed by a bicubic downscale; this mirrors that exactly.

Usage (needs torch, onnx, onnxruntime; CPU is fine):
    python scripts/export_onnx.py realesr-animevideov3.pth models/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN's compact VGG-style SR net (from basicsr)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def _bicubic_half_kernel(a: float = -0.75) -> torch.Tensor:
    """4-tap Keys cubic weights for an exact 2x downscale (sample offsets
    0.5 and 1.5 from the output pixel centre), same coefficient ncnn uses."""
    def w(d):
        d = abs(d)
        if d <= 1:
            return (a + 2) * d**3 - (a + 3) * d**2 + 1
        return a * d**3 - 5 * a * d**2 + 8 * a * d - 4 * a
    taps = torch.tensor([w(1.5), w(0.5), w(0.5), w(1.5)], dtype=torch.float32)
    return torch.outer(taps, taps)


class FrameModel(nn.Module):
    """uint8 NHWC in -> net (x4) -> optional downscale -> uint8 NHWC out."""

    def __init__(self, net: SRVGGNetCompact, out_scale: int, dtype: torch.dtype):
        super().__init__()
        self.net = net.to(dtype)
        self.out_scale = out_scale
        self.dtype = dtype
        k = _bicubic_half_kernel()
        self.register_buffer("down2", k.expand(3, 1, 4, 4).clone().to(dtype))

    def forward(self, frame):
        x = frame.permute(0, 3, 1, 2).to(self.dtype) / 255.0
        y = self.net(x)
        if self.out_scale == 2:
            y = F.pad(y, (1, 2, 1, 2), mode="replicate")
            y = F.conv2d(y, self.down2, stride=2, groups=3)
        elif self.out_scale == 3:
            y = F.interpolate(y, scale_factor=0.75, mode="bicubic", align_corners=False)
        y = torch.round(y.float().clamp(0, 1) * 255.0).to(torch.uint8)
        return y.permute(0, 2, 3, 1)


def main(pth: Path, out_dir: Path) -> None:
    state = torch.load(pth, map_location="cpu", weights_only=True)
    state = state.get("params_ema", state.get("params", state))
    net = SRVGGNetCompact()
    net.load_state_dict(state, strict=True)
    net.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    dummy = torch.randint(0, 256, (1, 64, 96, 3), dtype=torch.uint8)
    import onnxruntime as ort
    for scale in (2, 3, 4):
        import copy
        with torch.no_grad():
            reference = FrameModel(copy.deepcopy(net), scale, torch.float32).eval()(dummy).numpy()
        # Export straight from a half-precision module so the graph is fp16
        # end to end (weights and activations); IO stays uint8.
        model = FrameModel(copy.deepcopy(net), scale, torch.float16).eval()
        path = out_dir / f"realesr-animevideov3-x{scale}.onnx"
        with torch.no_grad():
            torch.onnx.export(
                model, (dummy,), str(path),
                input_names=["frame"], output_names=["upscaled"],
                dynamic_axes={"frame": {1: "height", 2: "width"}, "upscaled": {1: "out_height", 2: "out_width"}},
                opset_version=17, dynamo=False,
            )
        onnx.checker.check_model(onnx.load(str(path)))

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"frame": dummy.numpy()})[0]
        diff = np.abs(got.astype(np.int16) - reference.astype(np.int16))
        print(f"x{scale}: {path.name} {path.stat().st_size/1e6:.2f} MB, "
              f"out {got.shape}, max |diff| vs torch fp32 = {diff.max()}, mean = {diff.mean():.3f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
