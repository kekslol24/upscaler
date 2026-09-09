"""Export Real-ESRGAN's realesr-animevideov3 weights to ONNX for the
DirectML engine (backend/engines/realesrgan_onnx.py).

The graph takes a uint8 RGB frame (1, H, W, 3) and returns a uint8 RGB
frame, so the runtime can stream raw frames straight between ffmpeg and
the model without any float conversion in Python.

The network itself is a 4x model. ncnn's x2/x3 variants are that network
followed by a bicubic downscale of the 4x image. Doing that literally means
padding / resampling a 4x-resolution tensor (~200 MB at 1080p), and
DirectML truncates such Pad ops at 96 MiB (the bottom half of every x2
frame came back black). So here the "pixel-shuffle then bicubic-downscale"
pair — a fixed linear map — is folded into one small 3x3 convolution over
the network's 48-channel output at *input* resolution, followed by a 2x or
3x pixel shuffle. Same arithmetic, only ops the plain x4 graph already
uses at this size, and less memory traffic. The one difference: the
downscale's replicate border padding becomes zero padding, which changes
the outermost row/column of the output by a few percent.

Usage (needs torch, onnx, onnxruntime; CPU is fine):
    python scripts/export_onnx.py realesr-animevideov3.pth models/
    python scripts/export_onnx.py models/realesr-animevideov3-x4.onnx models/   # re-export from a previous export
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import numpy_helper


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

    def features(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return out

    def forward(self, x):
        out = self.upsampler(self.features(x))
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


def _reference_downscale(y4: torch.Tensor, out_scale: int) -> torch.Tensor:
    """What ncnn does to the 4x image to get x2 / x3 output (per channel)."""
    if out_scale == 2:
        k = _bicubic_half_kernel().expand(y4.shape[1], 1, 4, 4)
        return F.conv2d(F.pad(y4, (1, 2, 1, 2), mode="replicate"), k, stride=2, groups=y4.shape[1])
    if out_scale == 3:
        return F.interpolate(y4, scale_factor=0.75, mode="bicubic", align_corners=False)
    return y4


class FrameModel(nn.Module):
    """Literal reference: uint8 NHWC in -> net (x4) -> downscale -> uint8 NHWC out."""

    def __init__(self, net: SRVGGNetCompact, out_scale: int):
        super().__init__()
        self.net = net
        self.out_scale = out_scale

    def forward(self, frame):
        x = frame.permute(0, 3, 1, 2).float() / 255.0
        y = _reference_downscale(self.net(x), self.out_scale)
        y = torch.round(y.clamp(0, 1) * 255.0).to(torch.uint8)
        return y.permute(0, 2, 3, 1)


def fold_downscale_kernel(out_scale: int) -> torch.Tensor:
    """The linear map 'pixel-shuffle(4) then bicubic downscale to out_scale'
    as a 3x3 kernel over the 16 sub-pixel channels of one colour:
    (out_scale^2, 16, 3, 3). Derived by pushing unit impulses through the
    reference implementation on a canvas big enough to keep borders away."""
    s = out_scale
    kernel = torch.zeros(s * s, 16, 3, 3)
    for k in range(16):
        z = torch.zeros(1, 16, 5, 5)
        z[0, k, 2, 2] = 1.0
        y = _reference_downscale(F.pixel_shuffle(z, 4), s)  # (1, 1, 5s, 5s)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                # response at output block (2+di, 2+dj); conv2d is a correlation,
                # so that is the weight for the input at offset (-di, -dj)
                block = y[0, 0, s * (2 + di): s * (3 + di), s * (2 + dj): s * (3 + dj)]
                kernel[:, k, 1 - di, 1 - dj] = block.reshape(-1)
        leaked = y.clone()
        leaked[0, 0, s: 4 * s, s: 4 * s] = 0
        assert leaked.abs().max() < 1e-6, "downscale taps reach beyond neighbouring blocks"
    return kernel


class FoldedFrameModel(nn.Module):
    """uint8 NHWC in -> net body (48 ch at input res) -> folded downscale
    conv -> pixel shuffle(out_scale) -> uint8 NHWC out. No tensor larger
    than the network's own activations."""

    def __init__(self, net: SRVGGNetCompact, out_scale: int, dtype: torch.dtype):
        super().__init__()
        self.body = net.body.to(dtype)
        self.out_scale = out_scale
        self.dtype = dtype
        k = fold_downscale_kernel(out_scale)                      # (s*s, 16, 3, 3)
        # the nearest-4x skip connection is the input replicated into all
        # 16 sub-pixel channels, so its share of the downscale is the kernel
        # summed over those channels, applied to the input directly
        self.register_buffer("down_w", torch.cat([k] * 3, 0).to(dtype))           # (3*s*s, 16, 3, 3), groups=3
        self.register_buffer("skip_w", torch.cat([k.sum(1, keepdim=True)] * 3, 0).to(dtype))  # (3*s*s, 1, 3, 3)

    def forward(self, frame):
        x = frame.permute(0, 3, 1, 2).to(self.dtype) / 255.0
        feat = x
        for layer in self.body:
            feat = layer(feat)                                   # (1, 48, H, W)
        y = F.conv2d(feat, self.down_w, padding=1, groups=3) + F.conv2d(x, self.skip_w, padding=1, groups=3)
        y = F.pixel_shuffle(y, self.out_scale)
        y = torch.round(y.float().clamp(0, 1) * 255.0).to(torch.uint8)
        return y.permute(0, 2, 3, 1)


class PlainFrameModel(nn.Module):
    """x4: uint8 NHWC in -> net -> uint8 NHWC out (fp16 end to end)."""

    def __init__(self, net: SRVGGNetCompact, dtype: torch.dtype):
        super().__init__()
        self.net = net.to(dtype)
        self.dtype = dtype

    def forward(self, frame):
        x = frame.permute(0, 3, 1, 2).to(self.dtype) / 255.0
        y = torch.round(self.net(x).float().clamp(0, 1) * 255.0).to(torch.uint8)
        return y.permute(0, 2, 3, 1)


def load_state(path: Path) -> dict:
    if path.suffix == ".onnx":
        model = onnx.load(str(path))
        inits = {i.name: torch.from_numpy(numpy_helper.to_array(i).astype(np.float32)) for i in model.graph.initializer}
        state = {k[len("net."):]: v for k, v in inits.items() if k.startswith("net.body.")}
        # PReLU slopes lost their names in the export; they appear in layer order
        slopes = [n.input[1] for n in model.graph.node if n.op_type == "PRelu"]
        for i, name in enumerate(slopes):
            state[f"body.{2 * i + 1}.weight"] = inits[name].reshape(-1)
        return state
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state.get("params_ema", state.get("params", state))


def main(src: Path, out_dir: Path) -> None:
    net = SRVGGNetCompact()
    net.load_state_dict(load_state(src), strict=True)
    net.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    import onnxruntime as ort
    torch.manual_seed(0)
    dummy = torch.randint(0, 256, (1, 64, 96, 3), dtype=torch.uint8)
    for scale in (2, 3, 4):
        with torch.no_grad():
            reference = FrameModel(copy.deepcopy(net), scale).eval()(dummy).numpy()
            if scale == 4:
                model = PlainFrameModel(copy.deepcopy(net), torch.float16).eval()
            else:
                model = FoldedFrameModel(copy.deepcopy(net), scale, torch.float16).eval()
                folded32 = FoldedFrameModel(copy.deepcopy(net), scale, torch.float32).eval()(dummy).numpy()
                d = np.abs(folded32.astype(np.int16) - reference.astype(np.int16))
                print(f"x{scale}: folded vs reference (fp32): interior max |diff| = {d[0, 2:-2, 2:-2].max()}, "
                      f"border max |diff| = {d.max()}")
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
        ops = sorted({n.op_type for n in onnx.load(str(path)).graph.node})
        print(f"x{scale}: {path.name} {path.stat().st_size/1e6:.2f} MB, out {got.shape}, "
              f"max |diff| vs torch fp32 reference = {diff[0, 2:-2, 2:-2].max()} (interior), mean = {diff.mean():.3f}; "
              f"ops: {ops}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
