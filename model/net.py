"""Fully convolutional residual network with receptive field <= 3 cells per step."""
from __future__ import annotations

import torch
import torch.nn as nn

from model.normalize import ChannelNormalizer

NLAT = 32
NLON = 36
N_CHANNELS = 6
T_IN = 3
N_STATIC = 2
N_DOY = 2
IN_PLANES = T_IN * N_CHANNELS + N_STATIC + N_DOY  # 22


class _LocalResBlock(nn.Module):
    """Residual block with one 3x3 reflect-padded conv followed by a 1x1 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.conv3(x))
        h = self.act2(self.conv1(h))
        return x + h


class MonsoonNet(nn.Module):
    """Local fully convolutional residual predictor on a 32x36 grid.

    Uses exactly three 3x3 convolutions (receptive field radius = 3 cells per step)
    with reflect padding, no pooling, no strided/dilated convs, and no global ops.
    Static channels and normalization constants are registered buffers.
    """

    def __init__(
        self,
        static: torch.Tensor | None = None,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if static is None:
            static_t = torch.zeros((N_STATIC, NLAT, NLON), dtype=torch.float32)
        else:
            static_t = torch.as_tensor(static, dtype=torch.float32)
            if static_t.shape != (N_STATIC, NLAT, NLON):
                raise ValueError(f"Expected static shape {(N_STATIC, NLAT, NLON)}, got {tuple(static_t.shape)}")
        self.register_buffer("static", static_t)

        if mean is None:
            mean_t = torch.zeros(N_CHANNELS, dtype=torch.float32)
        else:
            mean_t = torch.as_tensor(mean, dtype=torch.float32)
        if std is None:
            std_t = torch.ones(N_CHANNELS, dtype=torch.float32)
        else:
            std_t = torch.as_tensor(std, dtype=torch.float32)

        self.norm = ChannelNormalizer(mean_t, std_t)
        self.hidden_dim = int(hidden_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(IN_PLANES, self.hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        # Exactly three 3x3 convolutions in total -> receptive field radius = 3 cells
        self.blocks = nn.Sequential(
            _LocalResBlock(self.hidden_dim),
            _LocalResBlock(self.hidden_dim),
            _LocalResBlock(self.hidden_dim),
        )
        self.head = nn.Conv2d(self.hidden_dim, N_CHANNELS, kernel_size=1)

    def forward(self, x_3days_norm: torch.Tensor, doy_sincos: torch.Tensor) -> torch.Tensor:
        """Predict normalized residual delta [B, 6, 32, 36] from [B, 3, 6, 32, 36] and [B, 2]."""
        bsz, t_in, n_ch, h, w = x_3days_norm.shape
        if t_in != T_IN or n_ch != N_CHANNELS or h != NLAT or w != NLON:
            raise ValueError(f"Expected input shape (B, {T_IN}, {N_CHANNELS}, {NLAT}, {NLON}), got {tuple(x_3days_norm.shape)}")

        x_flat = x_3days_norm.reshape(bsz, t_in * n_ch, h, w)
        static_planes = self.static.unsqueeze(0).expand(bsz, -1, -1, -1)
        doy_planes = doy_sincos.view(bsz, N_DOY, 1, 1).expand(bsz, N_DOY, h, w)
        inp = torch.cat([x_flat, static_planes, doy_planes], dim=1)

        feat = self.stem(inp)
        feat = self.blocks(feat)
        return self.head(feat)


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters())
