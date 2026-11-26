"""Latitude-weighted multi-step losses in normalized space."""
from __future__ import annotations

import torch
import torch.nn as nn

from monsoon_r1.contract import lat_weights
from monsoon_r1.scoring import RAIN_THRESH


class MonsoonLoss(nn.Module):
    """Combined latitude-weighted MSE, heavy-rain extra weight, gradient L1, and variance matching."""

    def __init__(
        self,
        w_mse: float = 1.0,
        w_rain_extra: float = 4.0,
        w_grad: float = 0.2,
        w_var: float = 0.1,
        rain_thresh: float = RAIN_THRESH,
        eps: float = 1e-6,
        channel_weights: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.w_mse = float(w_mse)
        self.w_rain_extra = float(w_rain_extra)
        self.w_grad = float(w_grad)
        self.w_var = float(w_var)
        self.rain_thresh = float(rain_thresh)
        self.eps = float(eps)

        w_lat = torch.tensor(lat_weights(), dtype=torch.float32)  # [32], mean == 1.0
        self.register_buffer("w_lat", w_lat)
        if channel_weights is None:
            w_chan = torch.ones((1, 1, 6, 1, 1), dtype=torch.float32)
        else:
            w_chan = torch.tensor(channel_weights, dtype=torch.float32).reshape(1, 1, 6, 1, 1)
        self.register_buffer("w_chan", w_chan)

    def forward(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        target_tp_phys: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total loss and individual unweighted term values.

        Args:
            pred_norm: [B, K, 6, 32, 36]
            target_norm: [B, K, 6, 32, 36]
            target_tp_phys: [B, K, 32, 36] physical precipitation in mm/day
        """
        w_5d = self.w_lat.view(1, 1, 1, -1, 1)
        w_4d = self.w_lat.view(1, 1, -1, 1)

        sq_err = (pred_norm - target_norm) ** 2
        loss_mse = (sq_err * w_5d * self.w_chan).mean()

        heavy_mask = (target_tp_phys > self.rain_thresh).float()
        loss_rain_extra = (sq_err[:, :, 1] * w_4d * heavy_mask).mean()

        dy_pred = pred_norm[..., 1:, :] - pred_norm[..., :-1, :]
        dy_true = target_norm[..., 1:, :] - target_norm[..., :-1, :]
        w_dy = 0.5 * (w_5d[..., 1:, :] + w_5d[..., :-1, :])

        dx_pred = pred_norm[..., :, 1:] - pred_norm[..., :, :-1]
        dx_true = target_norm[..., :, 1:] - target_norm[..., :, :-1]

        loss_grad = 0.5 * (
            (torch.abs(dy_pred - dy_true) * w_dy).mean()
            + (torch.abs(dx_pred - dx_true) * w_5d).mean()
        )

        # Latitude-weighted spatial variance per sample, per step, per channel
        mean_p = (pred_norm * w_5d).mean(dim=(-2, -1), keepdim=True)
        var_p = ((pred_norm - mean_p) ** 2 * w_5d).mean(dim=(-2, -1))
        mean_t = (target_norm * w_5d).mean(dim=(-2, -1), keepdim=True)
        var_t = ((target_norm - mean_t) ** 2 * w_5d).mean(dim=(-2, -1))
        loss_var = (torch.log((var_p + self.eps) / (var_t + self.eps)) ** 2).mean()

        total = (
            self.w_mse * loss_mse
            + self.w_rain_extra * loss_rain_extra
            + self.w_grad * loss_grad
            + self.w_var * loss_var
        )
        terms = {
            "mse": float(loss_mse.detach().item()),
            "rain_extra": float(loss_rain_extra.detach().item()),
            "grad_l1": float(loss_grad.detach().item()),
            "var_match": float(loss_var.detach().item()),
            "total": float(total.detach().item()),
        }
        return total, terms


def tcwv_conservation_loss(pred_phys: torch.Tensor, target_phys: torch.Tensor) -> torch.Tensor:
    """Prithvi-WxC inspired mass conservation penalty on global spatial sum of tcwv (channel 5)."""
    pred_sum = pred_phys[:, :, 5].sum(dim=(-2, -1))
    true_sum = target_phys[:, :, 5].sum(dim=(-2, -1))
    n_cells = float(pred_phys.shape[-2] * pred_phys.shape[-1])
    return torch.abs(pred_sum - true_sum).mean() / n_cells


def asymmetric_tp_variance_loss(
    pred_phys: torch.Tensor,
    target_phys: torch.Tensor,
    max_ratio: float = 2.0,
    min_ratio: float = 0.5,
    eps: float = 1e-4,
) -> torch.Tensor:
    """GenFocal-inspired asymmetric variance penalty on physical tp (channel 1)."""
    pred_tp = pred_phys[:, :, 1]
    true_tp = target_phys[:, :, 1]
    var_p = pred_tp.var(dim=(-2, -1), unbiased=False) + eps
    var_t = true_tp.var(dim=(-2, -1), unbiased=False) + eps
    log_ratio = torch.log(var_p / var_t)
    high_pen = torch.relu(log_ratio - torch.log(torch.tensor(max_ratio, device=pred_phys.device))) ** 2
    low_pen = torch.relu(torch.log(torch.tensor(min_ratio, device=pred_phys.device)) - log_ratio) ** 2
    return (high_pen + 0.25 * low_pen).mean()


def shock_propagation_loss(
    pred_shock_phys: torch.Tensor,
    pred_base_phys: torch.Tensor,
    near_masks: torch.Tensor,
    std_phys: torch.Tensor,
    target_out_frac: float = 0.12,
    max_step_growth: float = 2.0,
    max_z: float = 4.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute propagation, day-1 locality, and rollout stability losses for perturbed rollouts.

    Args:
        pred_shock_phys: [B, K, 6, 32, 36]
        pred_base_phys:  [B, K, 6, 32, 36]
        near_masks:      [B, 32, 36] bool mask of cells within 3 cells of shock box
        std_phys:        [6] physical channel std
    """
    diff = pred_shock_phys - pred_base_phys
    z = diff / std_phys.view(1, 1, 6, 1, 1)
    energy = (z ** 2).sum(dim=2)  # [B, K, 32, 36]

    near = near_masks.unsqueeze(1).float()  # [B, 1, 32, 36]
    far = 1.0 - near

    # Day-1 locality: penalize day-1 energy outside 3-cell neighborhood
    d1_tot = energy[:, 0].sum(dim=(-2, -1)) + eps
    d1_out = (energy[:, 0] * far[:, 0]).sum(dim=(-2, -1))
    loss_loc = (d1_out / d1_tot).mean()

    # Days 4..8 outward propagation: ensure >= target_out_frac of peak daily energy reaches ~near
    daily_tot = energy.sum(dim=(-2, -1))  # [B, K]
    peak_tot = daily_tot.max(dim=1).values.detach() + eps
    out_daily = (energy * far).sum(dim=(-2, -1))  # [B, K]
    late_out = out_daily[:, 3:8].mean(dim=1) if energy.shape[1] >= 8 else out_daily[:, -1]
    out_frac = late_out / peak_tot
    loss_prop = torch.relu(target_out_frac - out_frac).mean()

    # Stability: cap per-step energy growth at max_step_growth (2x) and bound max |z|
    if energy.shape[1] >= 2:
        step_ratio = (daily_tot[:, 1:] + eps) / (daily_tot[:, :-1].detach() + eps)
        loss_growth = (torch.relu(step_ratio - max_step_growth) ** 2).mean()
    else:
        loss_growth = torch.zeros((), device=pred_shock_phys.device, dtype=pred_shock_phys.dtype)
    loss_bound = (torch.relu(torch.abs(z) - max_z) ** 2).mean()
    loss_stab = loss_growth + loss_bound

    terms = {
        "shock_prop": float(loss_prop.detach().item()),
        "shock_loc": float(loss_loc.detach().item()),
        "shock_stab": float(loss_stab.detach().item()),
    }
    return loss_prop + 0.5 * loss_loc + 0.5 * loss_stab, terms


def channel_weighted_forecast_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    target_tp_phys: torch.Tensor,
    w_lat: torch.Tensor,
    channel_weights: torch.Tensor,
    w_mse: float = 1.0,
    w_rain_extra: float = 4.0,
    w_grad: float = 0.1,
    w_var: float = 0.1,
    w_tcwv_cons: float = 0.05,
    rain_thresh: float = RAIN_THRESH,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Latitude-weighted forecast loss with per-channel MSE weights and tcwv conservation."""
    w_5d = w_lat.view(1, 1, 1, -1, 1)
    w_4d = w_lat.view(1, 1, -1, 1)
    cw_5d = channel_weights.view(1, 1, 6, 1, 1)

    sq_err = (pred_norm - target_norm) ** 2
    loss_mse = (sq_err * w_5d * cw_5d).mean()

    heavy_mask = (target_tp_phys > rain_thresh).float()
    loss_rain_extra = (sq_err[:, :, 1] * w_4d * heavy_mask).mean()

    dy_pred = pred_norm[..., 1:, :] - pred_norm[..., :-1, :]
    dy_true = target_norm[..., 1:, :] - target_norm[..., :-1, :]
    w_dy = 0.5 * (w_5d[..., 1:, :] + w_5d[..., :-1, :])
    dx_pred = pred_norm[..., :, 1:] - pred_norm[..., :, :-1]
    dx_true = target_norm[..., :, 1:] - target_norm[..., :, :-1]
    loss_grad = 0.5 * (
        (torch.abs(dy_pred - dy_true) * w_dy).mean()
        + (torch.abs(dx_pred - dx_true) * w_5d).mean()
    )

    mean_p = (pred_norm * w_5d).mean(dim=(-2, -1), keepdim=True)
    var_p = ((pred_norm - mean_p) ** 2 * w_5d).mean(dim=(-2, -1))
    mean_t = (target_norm * w_5d).mean(dim=(-2, -1), keepdim=True)
    var_t = ((target_norm - mean_t) ** 2 * w_5d).mean(dim=(-2, -1))
    loss_var = (torch.log((var_p + eps) / (var_t + eps)) ** 2).mean()

    # Global spatial-mean consistency on tcwv (channel 5)
    tcwv_diff_mean = ((pred_norm[:, :, 5] - target_norm[:, :, 5]) * w_4d).mean(dim=(-2, -1))
    loss_tcwv_cons = (tcwv_diff_mean ** 2).mean()

    total = (
        w_mse * loss_mse
        + w_rain_extra * loss_rain_extra
        + w_grad * loss_grad
        + w_var * loss_var
        + w_tcwv_cons * loss_tcwv_cons
    )
    terms = {
        "mse": float(loss_mse.detach().item()),
        "rain_extra": float(loss_rain_extra.detach().item()),
        "grad_l1": float(loss_grad.detach().item()),
        "var_match": float(loss_var.detach().item()),
        "tcwv_cons": float(loss_tcwv_cons.detach().item()),
        "total": float(total.detach().item()),
    }
    return total, terms


def shock_causality_loss(
    base_phys: torch.Tensor,
    shocked_phys: torch.Tensor,
    near_masks: torch.Tensor,
    dist_grids: torch.Tensor,
    std_phys: torch.Tensor,
    w_local: float = 1.0,
    w_prop: float = 2.0,
    w_stable: float = 0.5,
    w_tele: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Differentiable locality, outward propagation, no-teleport, and stability loss on shock response."""
    diff = shocked_phys - base_phys  # [B, K, 6, 32, 36]
    z = diff / std_phys.view(1, 1, 6, 1, 1)
    energy = (z ** 2).sum(dim=2)  # [B, K, 32, 36]

    outside = 1.0 - near_masks  # [B, 32, 36]
    tot_d1 = energy[:, 0].sum(dim=(-2, -1)).clamp(min=1e-6)
    out_d1 = (energy[:, 0] * outside).sum(dim=(-2, -1))
    out_frac_d1 = out_d1 / tot_d1
    loss_local = torch.relu(out_frac_d1 - 0.35).mean()

    steps = energy.shape[1]
    out_per_step = (energy * outside.unsqueeze(1)).sum(dim=(-2, -1))  # [B, K]
    late_out = out_per_step[:, 3:8].mean(dim=1) if steps >= 8 else out_per_step[:, -1]
    peak_total = energy.sum(dim=(-2, -1)).max(dim=1).values.clamp(min=1e-6)

    # Encourage outward energy on days 4..8 >= 6% of peak total energy and >= 2x day-1 outside energy
    loss_prop = (
        torch.relu(0.06 - late_out / peak_total).mean()
        + torch.relu((2.0 * out_d1 - late_out) / peak_total).mean()
    )

    # Keep response amplitude safely below 5 std
    loss_stable = (torch.relu(z.abs() - 4.0) ** 2).mean()

    # Keep energy beyond 3 + 8*(k+1) cells small
    tele_losses = []
    for k in range(steps):
        far_mask = (dist_grids > (3.0 + 8.0 * (k + 1))).float()
        tot_k = energy[:, k].sum(dim=(-2, -1)).clamp(min=1e-6)
        far_frac = (energy[:, k] * far_mask).sum(dim=(-2, -1)) / tot_k
        tele_losses.append(torch.relu(far_frac - 0.03).mean())
    loss_tele = torch.stack(tele_losses).mean()

    total = (
        w_local * loss_local
        + w_prop * loss_prop
        + w_stable * loss_stable
        + w_tele * loss_tele
    )
    terms = {
        "shock_local": float(loss_local.detach().item()),
        "shock_prop": float(loss_prop.detach().item()),
        "shock_stable": float(loss_stable.detach().item()),
        "shock_tele": float(loss_tele.detach().item()),
        "shock_total": float(total.detach().item()),
    }
    return total, terms
