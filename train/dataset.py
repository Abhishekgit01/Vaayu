"""PyTorch dataset and training-split normalization statistics."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch.utils.data import Dataset

from monsoon_r1.contract import NC, T_IN


def compute_train_stats(train_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std from the training split (tp in log1p space)."""
    if train_data.ndim != 4 or train_data.shape[1] != NC:
        raise ValueError(f"Expected [T, {NC}, H, W], got {train_data.shape}")

    arr = train_data.astype(np.float32, copy=True)
    arr[:, 1] = np.log1p(np.maximum(arr[:, 1], 0.0))
    mean = arr.mean(axis=(0, 2, 3)).astype(np.float32)
    std = np.maximum(arr.std(axis=(0, 2, 3)), 1e-6).astype(np.float32)
    return mean, std


class MonsoonWindowDataset(Dataset):
    """Lazy sliding-window dataset over [T, 6, 32, 36] physical arrays."""

    def __init__(
        self,
        data: np.ndarray,
        doy: np.ndarray,
        t_in: int = T_IN,
        horizon: int = 8,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.doy = np.asarray(doy, dtype=np.int32)
        self.t_in = int(t_in)
        self.horizon = int(horizon)
        self.n_win = self.data.shape[0] - (self.t_in + self.horizon) + 1
        if self.n_win <= 0:
            raise ValueError(
                f"Split length {self.data.shape[0]} too short for t_in={self.t_in} + horizon={self.horizon}"
            )

    def set_horizon(self, horizon: int) -> None:
        """Update forecast horizon dynamically for curriculum training."""
        horizon = int(horizon)
        n_win = self.data.shape[0] - (self.t_in + horizon) + 1
        if n_win <= 0:
            raise ValueError(
                f"Split length {self.data.shape[0]} too short for t_in={self.t_in} + horizon={horizon}"
            )
        self.horizon = horizon
        self.n_win = n_win

    def __len__(self) -> int:
        return self.n_win

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        init = self.data[idx : idx + self.t_in]
        target = self.data[idx + self.t_in : idx + self.t_in + self.horizon]
        last_doy = float(self.doy[idx + self.t_in - 1])
        ang = 2.0 * math.pi * last_doy / 365.0
        doy_sc = np.array([math.sin(ang), math.cos(ang)], dtype=np.float32)
        return (
            torch.from_numpy(init),
            torch.from_numpy(doy_sc),
            torch.from_numpy(target),
        )
