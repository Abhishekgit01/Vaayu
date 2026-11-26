"""Per-channel normalization and de-normalization with log1p scaling for precipitation."""
from __future__ import annotations

import torch
import torch.nn as nn

TP_INDEX = 1


class ChannelNormalizer(nn.Module):
    """Normalizes and de-normalizes [..., 6, H, W] physical tensors using registered buffers."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(6)
        std_t = torch.as_tensor(std, dtype=torch.float32).reshape(6)
        if torch.any(std_t <= 0):
            raise ValueError("All channel std values must be strictly positive")
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Transform physical units -> normalized space (log1p on tp before z-scoring)."""
        out = x.clone()
        out[..., TP_INDEX, :, :] = torch.log1p(torch.clamp(out[..., TP_INDEX, :, :], min=0.0))
        view_shape = (1,) * (out.ndim - 3) + (6, 1, 1)
        return (out - self.mean.view(view_shape)) / self.std.view(view_shape)

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Transform normalized space -> physical units (expm1 on tp, clamped >= 0)."""
        view_shape = (1,) * (x_norm.ndim - 3) + (6, 1, 1)
        out = x_norm * self.std.view(view_shape) + self.mean.view(view_shape)
        out = out.clone()
        out[..., TP_INDEX, :, :] = torch.clamp(torch.expm1(out[..., TP_INDEX, :, :]), min=0.0)
        return out
