"""Batched differentiable rollout and validation-split evaluation table."""
from __future__ import annotations

import math
import numpy as np
import torch

from model.net import MonsoonNet
from monsoon_r1.contract import CHANNELS, LEAD_DAYS, T_IN
from monsoon_r1.scoring import BLUR_MAX_RATIO, BLUR_MIN_RATIO, RAIN_THRESH, csi, rmse_lat, variance_ratio
from train.baselines import fit_doy_climatology, predict_climatology, predict_persistence
from train.data import build_windows

T2M_MIN_K = 233.15
T2M_MAX_K = 333.15
_STEP_ANGLE = 2.0 * math.pi / 365.0
_COS_STEP = math.cos(_STEP_ANGLE)
_SIN_STEP = math.sin(_STEP_ANGLE)


def _normalize_out_of_place(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Out-of-place equivalent of model.norm.normalize for safe multi-step autograd."""
    tp = torch.log1p(torch.clamp(x[..., 1, :, :], min=0.0))
    x_log = torch.stack(
        [x[..., 0, :, :], tp, x[..., 2, :, :], x[..., 3, :, :], x[..., 4, :, :], x[..., 5, :, :]],
        dim=-3,
    )
    view_shape = (1,) * (x_log.ndim - 3) + (6, 1, 1)
    return (x_log - mean.view(view_shape)) / std.view(view_shape)


def _denormalize_out_of_place(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Out-of-place equivalent of model.norm.denormalize for safe multi-step autograd."""
    view_shape = (1,) * (x_norm.ndim - 3) + (6, 1, 1)
    x = x_norm * std.view(view_shape) + mean.view(view_shape)
    tp = torch.clamp(torch.expm1(x[..., 1, :, :]), min=0.0)
    return torch.stack(
        [x[..., 0, :, :], tp, x[..., 2, :, :], x[..., 3, :, :], x[..., 4, :, :], x[..., 5, :, :]],
        dim=-3,
    )


def rollout_differentiable(
    model: MonsoonNet,
    init_states: torch.Tensor,
    doy_sincos: torch.Tensor,
    steps: int,
    noise_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched differentiable rollout matching model/rollout.py physics and DOY rotation.

    Args:
        model: MonsoonNet
        init_states: [B, 3, 6, 32, 36] physical units
        doy_sincos: [B, 2] sin/cos of LAST input day
        steps: number of forecast steps
        noise_std: Gaussian noise std in normalized space added to model input when > 0

    Returns:
        preds_norm: [B, steps, 6, 32, 36]
        preds_phys: [B, steps, 6, 32, 36]
    """
    mean = model.norm.mean
    std = model.norm.std
    win_norm = _normalize_out_of_place(init_states, mean, std)
    cur_s = doy_sincos[:, 0].to(torch.float64)
    cur_c = doy_sincos[:, 1].to(torch.float64)

    preds_norm = []
    preds_phys = []

    for _ in range(int(steps)):
        model_in = win_norm
        if noise_std > 0.0 and model.training:
            model_in = win_norm + torch.randn_like(win_norm) * noise_std

        cur_doy = torch.stack([cur_s, cur_c], dim=-1).to(dtype=init_states.dtype)
        delta_norm = model(model_in, cur_doy)
        next_norm = win_norm[:, -1] + delta_norm

        next_phys = _denormalize_out_of_place(next_norm, mean, std)
        t2m_c = torch.clamp(next_phys[:, 0], min=T2M_MIN_K, max=T2M_MAX_K)
        tp_c = torch.clamp(next_phys[:, 1], min=0.0)
        next_phys_clamped = torch.stack(
            [t2m_c, tp_c, next_phys[:, 2], next_phys[:, 3], next_phys[:, 4], next_phys[:, 5]],
            dim=1,
        )

        preds_norm.append(next_norm)
        preds_phys.append(next_phys_clamped)

        next_norm_consistent = _normalize_out_of_place(next_phys_clamped, mean, std).unsqueeze(1)
        win_norm = torch.cat([win_norm[:, 1:], next_norm_consistent], dim=1)

        next_s = cur_s * _COS_STEP + cur_c * _SIN_STEP
        next_c = cur_c * _COS_STEP - cur_s * _SIN_STEP
        cur_s, cur_c = next_s, next_c

    return torch.stack(preds_norm, dim=1), torch.stack(preds_phys, dim=1)


def prepare_validation_references(
    train_data: np.ndarray,
    train_doy: np.ndarray,
    val_data: np.ndarray,
    val_doy: np.ndarray,
    max_windows: int | None = None,
) -> dict:
    """Precompute validation windows and baseline predictions once per run."""
    horizon = max(LEAD_DAYS)
    init_s, targets, doy_sc, target_doy = build_windows(val_data, val_doy, t_in=T_IN, horizon=horizon)
    if max_windows is not None and init_s.shape[0] > max_windows:
        init_s = init_s[:max_windows]
        targets = targets[:max_windows]
        doy_sc = doy_sc[:max_windows]
        target_doy = target_doy[:max_windows]

    clim = fit_doy_climatology(train_data, train_doy)
    pred_pers = predict_persistence(init_s, horizon=horizon)
    pred_clim = predict_climatology(clim, target_doy)

    tp_idx = CHANNELS.index("tp")
    refs = {}
    for lead in LEAD_DAYS:
        k = lead - 1
        y_true = targets[:, k]
        rmse_p = rmse_lat(pred_pers[:, k], y_true)
        rmse_c = rmse_lat(pred_clim[:, k], y_true)
        rmse_best = np.minimum(rmse_p, rmse_c)

        csi_p = csi(pred_pers[:, k, tp_idx], y_true[:, tp_idx], thr=RAIN_THRESH)
        csi_c = csi(pred_clim[:, k, tp_idx], y_true[:, tp_idx], thr=RAIN_THRESH)
        csi_best = float(np.nanmax([csi_p, csi_c])) if np.isfinite([csi_p, csi_c]).any() else float("nan")
        refs[lead] = {"rmse_best": rmse_best, "csi_best": csi_best}

    return {
        "init_states": init_s,
        "targets": targets,
        "doy_sincos": doy_sc,
        "refs": refs,
    }


def evaluate_model(
    model: MonsoonNet,
    val_refs: dict,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """Run model on validation windows and compute RMSE, skill, CSI, and day-8 variance ratio."""
    model.eval()
    init_s = val_refs["init_states"]
    targets = val_refs["targets"]
    doy_sc = val_refs["doy_sincos"]
    horizon = max(LEAD_DAYS)

    preds_list = []
    with torch.no_grad():
        for start in range(0, init_s.shape[0], batch_size):
            xb = torch.from_numpy(init_s[start : start + batch_size]).to(device)
            db = torch.from_numpy(doy_sc[start : start + batch_size]).to(device)
            _, pb = rollout_differentiable(model, xb, db, steps=horizon, noise_std=0.0)
            preds_list.append(pb.cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)

    tp_idx = CHANNELS.index("tp")
    lead_metrics = {}
    all_skills = []

    for lead in LEAD_DAYS:
        k = lead - 1
        y_pred = preds[:, k]
        y_true = targets[:, k]
        rmse_m = rmse_lat(y_pred, y_true)
        rmse_b = val_refs["refs"][lead]["rmse_best"]
        skill = np.where(rmse_b > 0, 1.0 - rmse_m / rmse_b, np.nan)
        all_skills.append(skill)

        csi_m = csi(y_pred[:, tp_idx], y_true[:, tp_idx], thr=RAIN_THRESH)
        csi_b = val_refs["refs"][lead]["csi_best"]
        lead_metrics[lead] = {
            "rmse_model": rmse_m,
            "rmse_better": rmse_b,
            "skill": skill,
            "csi_model": csi_m,
            "csi_better": csi_b,
        }

    # Day-8 spatial variance ratio per channel (averaged over validation windows)
    day8_ratios = np.stack(
        [variance_ratio(preds[i, 7], targets[i, 7]) for i in range(preds.shape[0])],
        axis=0,
    )
    mean_var_ratio = np.nanmean(day8_ratios, axis=0)

    mean_skill = float(np.nanmean(np.concatenate(all_skills)))
    var_penalty = 0.0
    for r in mean_var_ratio:
        if not np.isfinite(r):
            var_penalty += 1.0
        elif r < 0.5:
            var_penalty += 0.5 * (0.5 - float(r))
        elif r > 2.0:
            var_penalty += 0.5 * (float(r) - 2.0)

    selection_score = mean_skill - var_penalty
    return {
        "leads": lead_metrics,
        "day8_var_ratio": mean_var_ratio,
        "mean_skill": mean_skill,
        "selection_score": selection_score,
    }


def print_validation_table(eval_res: dict, epoch: int | None = None) -> None:
    """Print validation metrics in the same format as the baseline table."""
    hdr = f"Validation (epoch {epoch})" if epoch is not None else "Validation"
    print(f"\n{hdr} | mean_skill={eval_res['mean_skill']:+.4f} | score={eval_res['selection_score']:+.4f}")
    print("-" * 74)
    print(f"{'Lead':>4} | {'Channel':<6} | {'Model RMSE':>12} | {'Better Base':>12} | {'Skill':>10}")
    print("-" * 74)
    for lead in LEAD_DAYS:
        r = eval_res["leads"][lead]
        for ci, ch in enumerate(CHANNELS):
            print(
                f"{lead:>4} | {ch:<6} | {float(r['rmse_model'][ci]):>12.4f} | "
                f"{float(r['rmse_better'][ci]):>12.4f} | {float(r['skill'][ci]):>+10.4f}"
            )
        print("-" * 74)

    print("Heavy-rain CSI (tp > 10.0 mm/day):")
    print(f"{'Lead':>4} | {'Model CSI':>12} | {'Better Base':>12}")
    print("-" * 36)
    for lead in LEAD_DAYS:
        r = eval_res["leads"][lead]
        print(f"{lead:>4} | {r['csi_model']:>12.4f} | {r['csi_better']:>12.4f}")

    print("\nDay-8 spatial variance ratio (pred / true, target 0.40..2.50):")
    for ci, ch in enumerate(CHANNELS):
        vr = float(eval_res["day8_var_ratio"][ci])
        flag = " [OUTSIDE 0.4..2.5]" if (not np.isfinite(vr) or vr < BLUR_MIN_RATIO or vr > BLUR_MAX_RATIO) else ""
        print(f"  {ch:<6}: {vr:.4f}{flag}")
