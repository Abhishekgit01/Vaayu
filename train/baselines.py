"""Persistence and on-the-fly training-split climatology baselines."""
from __future__ import annotations

import numpy as np


def fit_doy_climatology(train_data: np.ndarray, train_doy: np.ndarray) -> dict[int, np.ndarray]:
    """Compute day-of-year mean climatology [6, 32, 36] on the fly from the training split.

    Never saved to disk. Falls back to overall training mean for any unseen DOY.
    """
    if train_data.shape[0] != train_doy.shape[0]:
        raise ValueError("train_data and train_doy length mismatch")

    global_mean = train_data.mean(axis=0).astype(np.float32)
    clim: dict[int, np.ndarray] = {0: global_mean}
    for d in range(1, 367):
        mask = train_doy == d
        if np.any(mask):
            clim[d] = train_data[mask].mean(axis=0).astype(np.float32)
        else:
            clim[d] = global_mean
    return clim


def predict_persistence(init_states: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last input day init_states[:, -1] for `horizon` steps."""
    last = init_states[:, -1]  # [N, 6, 32, 36]
    return np.repeat(last[:, None, ...], horizon, axis=1)


def predict_climatology(
    clim: dict[int, np.ndarray],
    target_doy: np.ndarray,
) -> np.ndarray:
    """Look up training-split DOY climatology for each [N, horizon] target day."""
    n_win, horizon = target_doy.shape
    sample = clim[0]
    out = np.empty((n_win, horizon, *sample.shape), dtype=np.float32)
    for i in range(n_win):
        for k in range(horizon):
            d = int(target_doy[i, k])
            out[i, k] = clim.get(d, clim[0])
    return out
