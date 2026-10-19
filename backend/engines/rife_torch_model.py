"""Minimal PyTorch implementation of the RIFE v4.6 inference network.

Vendored from hzwer/Practical-RIFE (MIT) so the CUDA interpolation path
does not depend on a separate checkout or the ncnn binaries.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_GRID_CACHE: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}


def _conv(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1,
          padding: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True),
    )


class _ResConv(nn.Module):
    def __init__(self, c: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class _IFBlock(nn.Module):
    def __init__(self, in_planes: int, c: int = 64) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            _conv(in_planes, c // 2, 3, 2, 1),
            _conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[_ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor, flow: torch.Tensor | None = None,
                scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(
                flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
            ) * (1.0 / scale)
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


def _warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    key = (str(flow.device), tuple(flow.size()))
    grid = _GRID_CACHE.get(key)
    if grid is None:
        horizontal = torch.linspace(
            -1.0, 1.0, flow.shape[3], device=flow.device
        ).view(1, 1, 1, flow.shape[3]).expand(flow.shape[0], -1, flow.shape[2], -1)
        vertical = torch.linspace(
            -1.0, 1.0, flow.shape[2], device=flow.device
        ).view(1, 1, flow.shape[2], 1).expand(flow.shape[0], -1, -1, flow.shape[3])
        grid = torch.cat([horizontal, vertical], 1).to(flow.device)
        _GRID_CACHE[key] = grid
    flow = torch.cat([
        flow[:, 0:1, :, :] / ((img.shape[3] - 1.0) / 2.0),
        flow[:, 1:2, :, :] / ((img.shape[2] - 1.0) / 2.0),
    ], 1)
    g = (grid + flow).permute(0, 2, 3, 1)
    return F.grid_sample(input=img, grid=g, mode="bilinear", padding_mode="border",
                         align_corners=True)


class RifeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = _IFBlock(7, c=192)
        self.block1 = _IFBlock(8 + 4, c=128)
        self.block2 = _IFBlock(8 + 4, c=96)
        self.block3 = _IFBlock(8 + 4, c=64)

    def forward(self, x: torch.Tensor, timestep: float = 0.5,
                scale_list: list[float] | tuple[float, ...] = (8, 4, 2, 1),
                ) -> tuple[list[torch.Tensor], torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]
        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        blocks = [self.block0, self.block1, self.block2, self.block3]
        for i in range(4):
            if flow is None:
                flow, mask = blocks[i](
                    torch.cat((img0[:, :3], img1[:, :3], timestep), 1), None,
                    scale=scale_list[i],
                )
            else:
                f0, m0 = blocks[i](
                    torch.cat((warped_img0[:, :3], warped_img1[:, :3], timestep, mask), 1),
                    flow, scale=scale_list[i],
                )
                flow = flow + f0
                mask = mask + m0
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = _warp(img0, flow[:, :2])
            warped_img1 = _warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        mask_list[3] = torch.sigmoid(mask_list[3])
        merged[3] = merged[3][0] * mask_list[3] + merged[3][1] * (1 - mask_list[3])
        return flow_list, mask_list[3], merged


class RifeModel:
    def __init__(self, weights: Path, device: torch.device) -> None:
        self.device = device
        self.net = RifeNet().to(device)
        self.net.eval()
        state = torch.load(str(weights), map_location=device)
        stripped = {
            key.replace("module.", ""): value
            for key, value in state.items()
            if key.startswith("module.")
        }
        self.net.load_state_dict(stripped or state)

    @torch.inference_mode()
    def infer(self, img0: torch.Tensor, img1: torch.Tensor,
              timestep: float = 0.5, scale: float = 1.0) -> torch.Tensor:
        scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        _, _, merged = self.net(torch.cat((img0, img1), 1), timestep, scale_list)
        return merged[3]
