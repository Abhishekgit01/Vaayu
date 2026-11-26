"""Autoregressive physical-unit rollout with exact DOY phase rotation."""
from __future__ import annotations

import math
import numpy as np
import torch

from model.net import MonsoonNet, N_CHANNELS, NLAT, NLON, T_IN

T2M_MIN_K = 233.15
T2M_MAX_K = 333.15
_STEP_ANGLE = 2.0 * math.pi / 365.0
_COS_STEP = math.cos(_STEP_ANGLE)
_SIN_STEP = math.sin(_STEP_ANGLE)


def advance_doy_sincos(doy_sincos: tuple[float, float] | np.ndarray, steps: int = 1) -> np.ndarray:
    """Rotate (sin, cos) by steps * (2*pi/365) using float64 precision."""
    s = float(doy_sincos[0])
    c = float(doy_sincos[1])
    ang = steps * _STEP_ANGLE
    cos_a = math.cos(ang)
    sin_a = math.sin(ang)
    s_new = s * cos_a + c * sin_a
    c_new = c * cos_a - s * sin_a
    return np.array([s_new, c_new], dtype=np.float32)


def rollout(
    model: MonsoonNet,
    init_states: np.ndarray,
    doy_sincos: tuple[float, float] | list[float] | np.ndarray,
    steps: int,
    perturbation: np.ndarray | None = None,
) -> np.ndarray:
    """Run deterministic autoregressive rollout in physical units.

    Args:
        model: MonsoonNet instance.
        init_states: [3, 6, 32, 36] array in physical units, oldest first.
        doy_sincos: [sin, cos] of the LAST input day (init_states[-1]).
        steps: number of forecast days to roll out.
        perturbation: optional [6, 32, 36] physical delta added to init_states[-1].

    Returns:
        np.ndarray of shape [steps, 6, 32, 36] (float32) in physical units.
    """
    init_arr = np.asarray(init_states, dtype=np.float32).copy()
    if init_arr.shape != (T_IN, N_CHANNELS, NLAT, NLON):
        raise ValueError(f"Expected init_states shape {(T_IN, N_CHANNELS, NLAT, NLON)}, got {init_arr.shape}")

    if perturbation is not None:
        pert_arr = np.asarray(perturbation, dtype=np.float32)
        if pert_arr.shape != (N_CHANNELS, NLAT, NLON):
            raise ValueError(f"Expected perturbation shape {(N_CHANNELS, NLAT, NLON)}, got {pert_arr.shape}")
        init_arr[-1] = init_arr[-1] + pert_arr
        init_arr[-1, 1] = np.maximum(init_arr[-1, 1], 0.0)

    device = next(model.parameters()).device
    model.eval()

    win_phys = torch.from_numpy(init_arr).unsqueeze(0).to(device=device, dtype=torch.float32)
    win_norm = model.norm.normalize(win_phys)

    cur_s = float(doy_sincos[0])
    cur_c = float(doy_sincos[1])
    preds = []

    with torch.no_grad():
        for _ in range(int(steps)):
            doy_t = torch.tensor([[cur_s, cur_c]], device=device, dtype=torch.float32)
            delta_norm = model(win_norm, doy_t)
            next_norm = win_norm[:, -1] + delta_norm

            next_phys = model.norm.denormalize(next_norm)
            next_phys[:, 0] = torch.clamp(next_phys[:, 0], min=T2M_MIN_K, max=T2M_MAX_K)
            next_phys[:, 1] = torch.clamp(next_phys[:, 1], min=0.0)

            preds.append(next_phys.squeeze(0).cpu().numpy())

            # Append predicted step to 3-day window and rotate DOY phase by 2*pi/365
            next_norm_consistent = model.norm.normalize(next_phys).unsqueeze(1)
            win_norm = torch.cat([win_norm[:, 1:], next_norm_consistent], dim=1)

            next_s = cur_s * _COS_STEP + cur_c * _SIN_STEP
            next_c = cur_c * _COS_STEP - cur_s * _SIN_STEP
            cur_s, cur_c = next_s, next_c

    out = np.stack(preds, axis=0).astype(np.float32)
    if not np.isfinite(out).all():
        raise FloatingPointError("Non-finite values produced during rollout")
    return out
