"""Held-out test-split skill, realistic causality, and aggressive-shock stability evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import serve
from monsoon_r1.contract import CHANNELS, LEAD_DAYS, T_IN
from monsoon_r1.scoring import (
    NLAT,
    NLON,
    RAIN_THRESH,
    STD_DEFAULT,
    box_mask,
    causality_checks,
    csi,
    rmse_lat,
    variance_ratio,
)
from train.baselines import fit_doy_climatology, predict_climatology, predict_persistence
from train.data import build_windows, compute_time_split, load_dataset


def _doy_to_sincos(doy: float) -> np.ndarray:
    ang = 2.0 * np.pi * float(doy) / 365.25
    return np.array([np.sin(ang), np.cos(ang)], dtype=np.float32)


def _footprint_radius_cells(diff: np.ndarray, box: tuple[float, float, float, float], thr_std: float = 0.05) -> list[float]:
    """Max Euclidean distance (grid cells) from shock box where ||d/std||_rms >= thr_std per day."""
    ii, jj = np.indices((NLAT, NLON))
    bi, bj = np.where(box_mask(box))
    dist = np.sqrt(((ii[..., None] - bi) ** 2 + (jj[..., None] - bj) ** 2)).min(axis=-1)
    z_rms = np.sqrt(np.mean((diff / STD_DEFAULT[None, :, None, None]) ** 2, axis=1))
    radii: list[float] = []
    for k in range(diff.shape[0]):
        active = z_rms[k] >= thr_std
        radii.append(float(dist[active].max()) if active.any() else 0.0)
    return radii


def run_phase2a(ds, splits, max_windows: int = 60) -> None:
    train_data = ds.data[splits.train]
    train_doy = ds.doy[splits.train]
    clim = fit_doy_climatology(train_data, train_doy)

    horizon = max(LEAD_DAYS)
    test_data = ds.data[splits.test]
    test_doy = ds.doy[splits.test]
    init_all, targets_all, init_doy_sc_all, target_doy_all = build_windows(
        test_data, test_doy, t_in=T_IN, horizon=horizon
    )
    n_avail = init_all.shape[0]
    n_eval = min(max_windows, n_avail)
    idx = np.linspace(0, n_avail - 1, n_eval, dtype=int)

    init_states = init_all[idx]
    targets = targets_all[idx]
    init_doy_sc = init_doy_sc_all[idx]
    target_doy = target_doy_all[idx]

    preds = np.empty_like(targets)
    for i in range(n_eval):
        mean_rollout, _ = serve.simulate_ensemble(
            init_states[i], init_doy_sc[i], steps=horizon, perturbation=None
        )
        preds[i] = mean_rollout

    pred_persist = predict_persistence(init_states, horizon=horizon)
    pred_clim = predict_climatology(clim, target_doy)

    print(f"\n=== PHASE 2A: TEST SPLIT SKILL (n_avail={n_avail}, n_eval={n_eval}, 5-member ensemble) ===")
    print(f"{'Lead':>4} | {'Ch':<4} | {'Model':>9} | {'Persist':>9} | {'Clim':>9} | {'BestBase':>9} | {'vsBest':<6}")
    print("-" * 68)
    for lead in LEAD_DAYS:
        k = lead - 1
        y_true = targets[:, k]
        rm_m = rmse_lat(preds[:, k], y_true)
        rm_p = rmse_lat(pred_persist[:, k], y_true)
        rm_c = rmse_lat(pred_clim[:, k], y_true)
        rm_b = np.minimum(rm_p, rm_c)
        for ci, ch in enumerate(CHANNELS):
            status = "WIN" if rm_m[ci] < rm_b[ci] else "LOSE"
            print(
                f"{lead:>4} | {ch:<4} | {rm_m[ci]:>9.4f} | {rm_p[ci]:>9.4f} | "
                f"{rm_c[ci]:>9.4f} | {rm_b[ci]:>9.4f} | {status:<6}"
            )
        print("-" * 68)

    tp_idx = CHANNELS.index("tp")
    print(f"\nHeavy-rain CSI (tp > {RAIN_THRESH} mm/day):")
    print(f"{'Lead':>4} | {'Model':>8} | {'Persist':>8} | {'Clim':>8} | {'BestBase':>8} | {'vsBest':<6}")
    print("-" * 56)
    for lead in LEAD_DAYS:
        k = lead - 1
        c_m = csi(preds[:, k, tp_idx], targets[:, k, tp_idx], thr=RAIN_THRESH)
        c_p = csi(pred_persist[:, k, tp_idx], targets[:, k, tp_idx], thr=RAIN_THRESH)
        c_c = csi(pred_clim[:, k, tp_idx], targets[:, k, tp_idx], thr=RAIN_THRESH)
        c_b = float(np.nanmax([c_p, c_c]))
        status = "WIN" if c_m > c_b else "LOSE"
        print(f"{lead:>4} | {c_m:>8.4f} | {c_p:>8.4f} | {c_c:>8.4f} | {c_b:>8.4f} | {status:<6}")

    # Day-8 spatial variance ratio (mean across windows of per-window variance_ratio, or pooled)
    vr_list = np.array([variance_ratio(preds[i, 7], targets[i, 7]) for i in range(n_eval)])
    vr_mean = np.nanmean(vr_list, axis=0)
    print("\nDay-8 Variance Ratios (pred / true spatial variance, mean over 60 test windows):")
    for ci, ch in enumerate(CHANNELS):
        in_band = 0.40 <= vr_mean[ci] <= 2.50
        print(f"  {ch:<4}: {vr_mean[ci]:.4f}  ({'OK [0.40, 2.50]' if in_band else 'OUTSIDE'})")


def run_phase2b(ds, splits) -> None:
    test_data = ds.data[splits.test]
    test_doy = ds.doy[splits.test]
    init_all, _, init_doy_sc_all, _ = build_windows(test_data, test_doy, t_in=T_IN, horizon=8)
    last_doy = test_doy[T_IN - 1 : T_IN - 1 + len(init_all)]
    monsoon_cands = np.where((last_doy >= 170) & (last_doy <= 240))[0]
    w_idx = int(monsoon_cands[len(monsoon_cands) // 2]) if len(monsoon_cands) else len(init_all) // 2
    init = init_all[w_idx]
    doy_sc = init_doy_sc_all[w_idx]

    base, _ = serve.simulate_ensemble(init, doy_sc, steps=8, perturbation=None)
    bob_box = (8.0, 20.0, 82.0, 94.0)
    arab_box = (8.0, 20.0, 55.0, 75.0)
    bob_mask = box_mask(bob_box)
    arab_mask = box_mask(arab_box)

    scenarios = []
    for dt in (2.0, 3.0, 4.0):
        pert = np.zeros((6, NLAT, NLON), dtype=np.float32)
        pert[0, bob_mask] = dt
        pert[5, bob_mask] = 0.15 * init[-1, 5, bob_mask]
        scenarios.append((f"BoB +{dt:.0f}K t2m, +15% tcwv (8-20N,82-94E)", bob_box, pert))

    pert_u = np.zeros((6, NLAT, NLON), dtype=np.float32)
    pert_u[2, arab_mask] = -0.40 * init[-1, 2, arab_mask]
    scenarios.append(("Arabian -40% u10 (8-20N,55-75E)", arab_box, pert_u))

    print(f"\n=== PHASE 2B: REALISTIC CAUSALITY CHECKS (test window idx={w_idx}, doy={int(last_doy[w_idx])}) ===")
    print(f"{'Scenario':<42} | {'near_d1':>7} | {'far_max':>7} | {'prop':<5} | {'Footprint radius D1..D8 (cells, >=0.05 std)'}")
    print("-" * 120)
    for label, box, pert in scenarios:
        shock, _ = serve.simulate_ensemble(init, doy_sc, steps=8, perturbation=pert)
        res = causality_checks(base, shock, box, STD_DEFAULT)
        radii = _footprint_radius_cells(shock - base, box, thr_std=0.05)
        r_str = " ".join(f"{r:4.1f}" for r in radii)
        print(
            f"{label:<42} | {res['near_frac_day1']:>7.4f} | {res['far_frac_max']:>7.4f} | "
            f"{str(res['propagates']):<5} | {r_str}"
        )


def run_phase2c(ds, splits) -> None:
    test_data = ds.data[splits.test]
    test_doy = ds.doy[splits.test]
    init_all, _, init_doy_sc_all, _ = build_windows(test_data, test_doy, t_in=T_IN, horizon=12)
    last_doy = test_doy[T_IN - 1 : T_IN - 1 + len(init_all)]

    def _pick(lo: int, hi: int, fallback: int) -> int:
        c = np.where((last_doy >= lo) & (last_doy <= hi))[0]
        return int(c[len(c) // 2]) if len(c) else fallback

    seasons = [
        ("Winter (DJF)", _pick(1, 50, 0)),
        ("Pre-monsoon (MAM)", _pick(90, 145, len(init_all) // 3)),
        ("Monsoon (JJAS)", _pick(180, 240, (2 * len(init_all)) // 3)),
    ]

    bob_mask = box_mask((8.0, 20.0, 82.0, 94.0))
    arab_mask = box_mask((8.0, 20.0, 55.0, 75.0))
    pert_t = np.zeros((6, NLAT, NLON), dtype=np.float32)
    pert_t[0, bob_mask] = 10.0
    pert_u = np.zeros((6, NLAT, NLON), dtype=np.float32)
    pert_u[2, arab_mask] = -20.0

    print("\n=== PHASE 2C: AGGRESSIVE SHOCK STABILITY (12 steps, 3 seasons) ===")
    print(f"{'Season':<20} | {'Shock':<18} | {'finite':<6} | {'tp>=0':<5} | {'t2m_min (K)':>11} | {'t2m_max (K)':>11}")
    print("-" * 82)
    for s_name, w_idx in seasons:
        init = init_all[w_idx]
        doy_sc = init_doy_sc_all[w_idx]
        for sh_name, pert in (("+10 K t2m box", pert_t), ("-20 m/s u10 box", pert_u)):
            out, _ = serve.simulate_ensemble(init, doy_sc, steps=12, perturbation=pert)
            fin = bool(np.isfinite(out).all())
            tp_ok = bool((out[:, 1] >= 0.0).all())
            t2m_min = float(out[:, 0].min())
            t2m_max = float(out[:, 0].max())
            print(f"{s_name:<20} | {sh_name:<18} | {str(fin):<6} | {str(tp_ok):<5} | {t2m_min:>11.2f} | {t2m_max:>11.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out test split, realistic causality, and extreme shocks.")
    parser.add_argument("--dataset", required=True, help="Path to india_climate_train.nc")
    args = parser.parse_args()
    ds = load_dataset(args.dataset)
    splits = compute_time_split(ds.data.shape[0])
    run_phase2a(ds, splits)
    run_phase2b(ds, splits)
    run_phase2c(ds, splits)


if __name__ == "__main__":
    main()
