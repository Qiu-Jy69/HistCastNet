# Derived from Earthformer (Apache License 2.0).
# Modified for HistCastNet.
"""Compatibility modules for HistCastNet's independent 2-D Haar transform."""

from __future__ import annotations

import torch
from torch import nn

from .haar import haar_dwt2d, haar_idwt2d

__all__ = ["DWT_2D", "IDWT_2D", "DWT_2D_tiny", "FrameWiseDWT2D", "FrameWiseIDWT2D"]


def _require_haar(wavename: str) -> None:
    if wavename.lower() != "haar":
        raise ValueError("HistCastNet supports only the Haar wavelet.")


class DWT_2D(nn.Module):
    """Historical DWT interface backed by the independent Haar definition."""

    def __init__(self, wavename: str):
        super().__init__()
        _require_haar(wavename)

    def forward(self, input: torch.Tensor):
        return haar_dwt2d(input)


class DWT_2D_tiny(DWT_2D):
    """Return only the LL Haar subband, preserving the former interface."""

    def forward(self, input: torch.Tensor):
        return haar_dwt2d(input)[0]


class IDWT_2D(nn.Module):
    """Historical IDWT interface backed by the independent Haar definition."""

    def __init__(self, wavename: str):
        super().__init__()
        _require_haar(wavename)

    def forward(self, LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor):
        return haar_idwt2d(LL, LH, HL, HH)


class FrameWiseDWT2D(nn.Module):
    def __init__(self, wavename: str = "haar"):
        super().__init__()
        self.dwt2d = DWT_2D(wavename)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(-1)
        if x.dim() != 5:
            raise ValueError("Expected (B, T, H, W[, C]) input.")
        batch, steps, height, width, channels = x.shape
        x_2d = x.permute(0, 1, 4, 2, 3).reshape(batch * steps, channels, height, width)
        ll, lh, hl, hh = self.dwt2d(x_2d)
        x_wavelet = torch.cat((ll, lh, hl, hh), dim=1)
        _, wavelet_channels, height_half, width_half = x_wavelet.shape
        return x_wavelet.reshape(batch, steps, wavelet_channels, height_half, width_half).permute(0, 1, 3, 4, 2).contiguous()


class FrameWiseIDWT2D(nn.Module):
    def __init__(self, wavename: str = "haar"):
        super().__init__()
        self.idwt2d = IDWT_2D(wavename)

    def forward(self, x_dwt: torch.Tensor) -> torch.Tensor:
        if x_dwt.dim() != 5:
            raise ValueError("Expected (B, T, H, W, 4C) input.")
        batch, steps, height_half, width_half, wavelet_channels = x_dwt.shape
        if wavelet_channels % 4:
            raise ValueError("The channel dimension must be divisible by four.")
        channels = wavelet_channels // 4
        x = x_dwt.permute(0, 1, 4, 2, 3).reshape(batch * steps, wavelet_channels, height_half, width_half)
        ll, lh, hl, hh = x.split(channels, dim=1)
        reconstruction = self.idwt2d(ll, lh, hl, hh)
        _, output_channels, height, width = reconstruction.shape
        return reconstruction.reshape(batch, steps, output_channels, height, width).permute(0, 1, 3, 4, 2).contiguous()
