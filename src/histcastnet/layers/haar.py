"""Pure-PyTorch two-dimensional Haar wavelet transforms for HistCastNet.

The implementation follows the Haar basis definition directly.  It has no
trainable parameters and uses ordinary PyTorch tensor operations, so autograd
is provided by PyTorch.
"""

from __future__ import annotations

import math

import torch


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _analysis_axis(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the Haar analysis basis along ``dim``.

    For an odd-length axis, this preserves the historical boundary convention:
    the final high-pass coefficient contains the final sample scaled by
    ``1 / sqrt(2)``.
    """
    size = x.size(dim)
    pairs = size // 2
    even = x.narrow(dim, 0, pairs * 2).index_select(
        dim, torch.arange(0, pairs * 2, 2, device=x.device)
    )
    odd = x.narrow(dim, 0, pairs * 2).index_select(
        dim, torch.arange(1, pairs * 2, 2, device=x.device)
    )
    low = (even + odd) * _INV_SQRT2
    high = (even - odd) * _INV_SQRT2
    if size % 2:
        high = torch.cat((high, x.narrow(dim, size - 1, 1) * _INV_SQRT2), dim=dim)
    return low, high


def _synthesis_axis(low: torch.Tensor, high: torch.Tensor, dim: int) -> torch.Tensor:
    """Apply the transpose of the historical Haar analysis basis along ``dim``."""
    pairs = low.size(dim)
    output_size = low.size(dim) + high.size(dim)
    shape = list(low.shape)
    shape[dim] = output_size
    output = low.new_zeros(shape)
    even_index = torch.arange(0, pairs * 2, 2, device=low.device)
    odd_index = torch.arange(1, pairs * 2, 2, device=low.device)
    output.index_copy_(dim, even_index, (low + high.narrow(dim, 0, pairs)) * _INV_SQRT2)
    output.index_copy_(dim, odd_index, (low - high.narrow(dim, 0, pairs)) * _INV_SQRT2)
    if high.size(dim) > pairs:
        output.index_copy_(dim, torch.tensor([output_size - 1], device=low.device), high.narrow(dim, pairs, 1) * _INV_SQRT2)
    return output


def haar_dwt2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return 2-D Haar subbands in the historical LL, LH, HL, HH order.

    Input layout is ``(N, C, H, W)``.  The signs and normalization match the
    former matrix implementation used by HistCastNet.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected a 4-D (N, C, H, W) tensor, got {x.ndim} dimensions.")
    low_rows, high_rows = _analysis_axis(x, -2)
    ll, lh = _analysis_axis(low_rows, -1)
    hl, hh = _analysis_axis(high_rows, -1)
    return ll, lh, hl, hh


def haar_idwt2d(
    ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor
) -> torch.Tensor:
    """Reconstruct a 2-D tensor from historical-order Haar subbands."""
    if not (ll.ndim == lh.ndim == hl.ndim == hh.ndim == 4):
        raise ValueError("All Haar subbands must be 4-D (N, C, H, W) tensors.")
    low_rows = _synthesis_axis(ll, lh, -1)
    high_rows = _synthesis_axis(hl, hh, -1)
    return _synthesis_axis(low_rows, high_rows, -2)
