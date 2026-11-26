"""Read-only diagnostics for Shock Causality (B), Training/Loss Scaling (C), and Ensemble Sweep (D)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import serve
from model.rollout import rollout as model_rollout
from monsoon_r1.contract import CHANNELS, LAT, LEAD_DAYS, LON, T_IN, lat_weights
from monsoon_r1.scoring import (
    DILATE,
    NLAT,
    NLON,
    RAIN_THRESH,
    STD_DEFAULT,
    _energy,
    box_mask,
    causality_checks,
    csi,
    near_mask,
    rmse_lat,
    variance_ratio,
)
from train.baselines import fit_doy_climatology, predict_climatology, predict_persistence
from train.data import build_windows, compute_time_split, load_dataset
from train.losses import MonsoonLoss, channel_weighted_forecast_loss


def _pick_season_windows(doy_arr: np.ndarray) -> list[tuple[str, int]]:
    def _pick(lo: int, hi: int, fb: int) -> int:
        c = np.where((doy_arr >= lo) & (doy_arr <= hi))[0]
        return int(c[len(c) // 2]) if len(c) else fb

    return [
        ("Winter(DJF)", _pick(1, 50, 0)),
        ("PreMon(MAM)", _pick(90, 145, len(doy_arr) // 3)),
        ("Monsoon(JJAS)", 423 if 423 < len(doy_arr) else _pick(165, 240, (2 * len(doy_arr)) // 3)),
    ]


def run_b(ds, splits) -> None:
    test_data = ds.data[splits.test]
    test_doy = ds.doy[splits.test]
    init_all, _, init_doy_sc_all, _ = build_windows(test_data, test_doy, t_in=T_IN, horizon=12)
    last_doy = test_doy[T_IN - 1 : T_IN - 1 + len(init_all)]
    seasons = _pick_season_windows(last_doy)

    bob_box = (8.0, 20.0, 82.0, 94.0)
    arab_box = (8.0, 20.0, 55.0, 75.0)
    bob_mask = box_mask(bob_box)
    arab_mask = box_mask(arab_box)

    print("=== TASK B.1 & B.2 & B.3: SHOCK DIAGNOSTICS (3 SEASONS ON TEST SPLIT) ===")
    print("Kit propagates logic: late = out_e[3:8].mean(); grows = (late > 1.5*out_e[0] + 1e-12) and (late > 0.02*e_peak); tele < 0.05")

    for s_name, w_idx in seasons:
        init = init_all[w_idx]
        doy_sc = init_doy_sc_all[w_idx]
        base, _ = serve.simulate_ensemble(init, doy_sc, steps=8, perturbation=None)

        print(f"\n--- Season: {s_name} (test idx={w_idx}, doy={int(last_doy[w_idx])}) ---")
        bob_anoms = {}
        for dt in (2.0, 3.0, 4.0):
            pert = np.zeros((6, NLAT, NLON), dtype=np.float32)
            pert[0, bob_mask] = dt
            pert[5, bob_mask] = 0.15 * init[-1, 5, bob_mask]
            shock, _ = serve.simulate_ensemble(init, doy_sc, steps=8, perturbation=pert)
            res = causality_checks(base, shock, bob_box, STD_DEFAULT)
            d = shock - base
            e = _energy(d, STD_DEFAULT)
            near = near_mask(bob_box)
            out_e = np.array([e[k][~near].sum() for k in range(8)])
            e_peak = max(float(e.sum(axis=(1, 2)).max()), 1e-12)
            late = float(out_e[3:8].mean())
            ratios = " ".join(f"{out_e[k]/e_peak:.4f}" for k in range(8))
            print(
                f"  BoB +{dt:.0f}K: near_d1={res['near_frac_day1']:.4f} far_max={res['far_frac_max']:.4f} "
                f"prop={res['propagates']} | late/peak={late/e_peak:.4f} (thr 0.02) "
                f"late/out_d1={late/max(out_e[0],1e-12):.2f}x | out_e/peak D1..D8: {ratios}"
            )
            in_mean = [float(d[k, 0, bob_mask].mean()) for k in (0, 1, 3, 7)]
            out_max = [float(np.abs(d[k, 0, ~bob_mask]).max()) for k in (0, 1, 3, 7)]
            bob_anoms[dt] = (in_mean, out_max)
            print(
                f"    t2m anom in-box mean D1/D2/D4/D8: {[round(x, 4) for x in in_mean]} | "
                f"out-box max|d| D1/D2/D4/D8: {[round(x, 4) for x in out_max]}"
            )
        r42 = [round(bob_anoms[4.0][0][i] / max(bob_anoms[2.0][0][i], 1e-9), 3) for i in range(4)]
        print(f"  BoB 4K/2K in-box t2m ratio D1/D2/D4/D8: {r42}")

        pert_u = np.zeros((6, NLAT, NLON), dtype=np.float32)
        pert_u[2, arab_mask] = -0.40 * init[-1, 2, arab_mask]
        shock_u, _ = serve.simulate_ensemble(init, doy_sc, steps=8, perturbation=pert_u)
        res_u = causality_checks(base, shock_u, arab_box, STD_DEFAULT)
        d_u = shock_u - base
        e_u = _energy(d_u, STD_DEFAULT)
        near_u = near_mask(arab_box)
        out_eu = np.array([e_u[k][~near_u].sum() for k in range(8)])
        e_peaku = max(float(e_u.sum(axis=(1, 2)).max()), 1e-12)
        late_u = float(out_eu[3:8].mean())
        ratios_u = " ".join(f"{out_eu[k]/e_peaku:.4f}" for k in range(8))
        u_in = [float(d_u[k, 2, arab_mask].mean()) for k in (0, 1, 3, 7)]
        u_out = [float(np.abs(d_u[k, 2, ~arab_mask]).max()) for k in (0, 1, 3, 7)]
        print(
            f"  Arab -40% u10: near_d1={res_u['near_frac_day1']:.4f} far_max={res_u['far_frac_max']:.4f} "
            f"prop={res_u['propagates']} | late/peak={late_u/e_peaku:.4f} | out_e/peak D1..D8: {ratios_u}"
        )
        print(
            f"    u10 anom in-box mean D1/D2/D4/D8: {[round(x, 4) for x in u_in]} | "
            f"out-box max|d| D1/D2/D4/D8: {[round(x, 4) for x in u_out]}"
        )

    print("\n=== TASK B.4: PHASE 2C WITH NO-SHOCK CONTROL (12 STEPS, 3 SEASONS) ===")
    print(f"{'Season':<14} | {'Case':<16} | {'finite':<6} | {'tp>=0':<5} | {'t2m_max(K)':>10} | {'max cell (r,c) & (lat,lon)'}")
    print("-" * 88)
    pert_t = np.zeros((6, NLAT, NLON), dtype=np.float32)
    pert_t[0, bob_mask] = 10.0
    pert_u2 = np.zeros((6, NLAT, NLON), dtype=np.float32)
    pert_u2[2, arab_mask] = -20.0
    for s_name, w_idx in seasons:
        init = init_all[w_idx]
        doy_sc = init_doy_sc_all[w_idx]
        for label, pert in (("Control (null)", None), ("+10 K t2m box", pert_t), ("-20 m/s u10 box", pert_u2)):
            out, _ = serve.simulate_ensemble(init, doy_sc, steps=12, perturbation=pert)
            t2m = out[:, 0]
            flat_idx = int(np.argmax(t2m))
            step_m, r_m, c_m = np.unravel_index(flat_idx, t2m.shape)
            print(
                f"{s_name:<14} | {label:<16} | {str(bool(np.isfinite(out).all())):<6} | "
                f"{str(bool((out[:,1]>=0).all())):<5} | {float(t2m.max()):>10.2f} | "
                f"d={step_m+1} ({r_m},{c_m}) ({LAT[r_m]:.1f}N,{LON[c_m]:.1f}E)"
            )


def run_c(ds, splits) -> None:
    train_data = ds.data[splits.train]
    val_data = ds.data[splits.val]
    val_doy = ds.doy[splits.val]

    state_std = np.std(train_data, axis=(0, 2, 3))
    delta_std = np.std(train_data[1:] - train_data[:-1], axis=(0, 2, 3))
    print("\n=== TASK C: TARGET SCALING & LOSS TERM BREAKDOWN ON VAL SPLIT ===")
    print("Target scaling: model predicts delta_norm = (x_{t+1} - x_t) / state_std (normalized by STATE std, not delta std).")
    print(f"{'Ch':<5} | {'state_std':>10} | {'delta_std':>10} | {'delta/state':>11}")
    for ci, ch in enumerate(CHANNELS):
        print(f"{ch:<5} | {state_std[ci]:>10.4f} | {delta_std[ci]:>10.4f} | {delta_std[ci]/state_std[ci]:>11.4f}")

    init_v, targ_v, doy_sc_v, _ = build_windows(val_data, val_doy, t_in=T_IN, horizon=8)
    idx = np.linspace(0, len(init_v) - 1, min(60, len(init_v)), dtype=int)
    init_v, targ_v, doy_sc_v = init_v[idx], targ_v[idx], doy_sc_v[idx]

    preds = np.empty_like(targ_v)
    for i in range(len(idx)):
        preds[i] = model_rollout(serve._MODEL, init_v[i], doy_sc_v[i], steps=8, perturbation=None)

    # Per-horizon val MSE in normalized space & t2m D1 RMSE vs persistence
    pred_norm = serve._MODEL.norm.normalize(torch.from_numpy(preds))
    targ_norm = serve._MODEL.norm.normalize(torch.from_numpy(targ_v))
    w_lat = torch.tensor(lat_weights(), dtype=torch.float32)
    w_5d = w_lat.view(1, 1, 1, -1, 1)

    print("\nPer-horizon val normalized MSE (1..8):")
    for k in range(8):
        mse_k = float((((pred_norm[:, k] - targ_norm[:, k]) ** 2) * w_5d.squeeze(1)).mean().item())
        print(f"  Day {k+1}: norm_mse={mse_k:.4f}")

    pers = predict_persistence(init_v, horizon=8)
    t2m_d1_model = float(rmse_lat(preds[:, 0], targ_v[:, 0])[0])
    t2m_d1_pers = float(rmse_lat(pers[:, 0], targ_v[:, 0])[0])
    print(f"\nVal t2m Day-1 RMSE: model={t2m_d1_model:.4f} K vs persistence={t2m_d1_pers:.4f} K")

    # Loss term shares under MonsoonLoss (train.py defaults) and channel_weighted_forecast_loss (train_shock.py defaults)
    crit_base = MonsoonLoss(w_mse=1.0, w_rain_extra=4.0, w_grad=0.2, w_var=0.1)
    tot_b, terms_b = crit_base(pred_norm, targ_norm, torch.from_numpy(targ_v[:, :, 1]))
    w_b = {
        "mse": 1.0 * terms_b["mse"],
        "rain_extra": 4.0 * terms_b["rain_extra"],
        "grad_l1": 0.2 * terms_b["grad_l1"],
        "var_match": 0.1 * terms_b["var_match"],
    }
    print("\nMonsoonLoss (train.py defaults: w_mse=1.0, w_rain_extra=4.0, w_grad=0.2, w_var=0.1):")
    for k, v in w_b.items():
        print(f"  {k:<12}: unweighted={terms_b[k]:.4f} | weighted={v:.4f} ({100*v/terms_b['total']:.1f}% of total={terms_b['total']:.4f})")

    cw = torch.tensor([2.5, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    tot_s, terms_s = channel_weighted_forecast_loss(
        pred_norm, targ_norm, torch.from_numpy(targ_v[:, :, 1]),
        w_lat=w_lat, channel_weights=cw, w_mse=1.0, w_rain_extra=4.0, w_grad=0.1, w_var=0.1, w_tcwv_cons=0.05
    )
    w_s = {
        "mse": 1.0 * terms_s["mse"],
        "rain_extra": 4.0 * terms_s["rain_extra"],
        "grad_l1": 0.1 * terms_s["grad_l1"],
        "var_match": 0.1 * terms_s["var_match"],
        "tcwv_cons": 0.05 * terms_s["tcwv_cons"],
    }
    print("\nchannel_weighted_forecast_loss (train_shock.py defaults: w_t2m=2.5, w_grad=0.1, w_var=0.1, w_tcwv_cons=0.05):")
    for k, v in w_s.items():
        print(f"  {k:<12}: unweighted={terms_s[k]:.4f} | weighted={v:.4f} ({100*v/terms_s['total']:.1f}% of total={terms_s['total']:.4f})")


def _sim_custom_ensemble(init_states: np.ndarray, doy_sc: np.ndarray, steps: int, pert: np.ndarray | None, noise_scale: float, n_members: int) -> np.ndarray:
    base_init = np.asarray(init_states, dtype=np.float32).copy()
    if pert is not None:
        base_init[-1] = base_init[-1] + np.asarray(pert, dtype=np.float32)
        base_init[-1, 1] = np.maximum(base_init[-1, 1], 0.0)
    rng = np.random.default_rng(42)
    std_arr = np.asarray(serve.CHANNEL_STD_PHYS, dtype=np.float32).reshape(1, 6, 1, 1)
    members = [model_rollout(serve._MODEL, base_init, doy_sc, steps=steps, perturbation=None)]
    for _ in range(n_members - 1):
        m_init = base_init.copy()
        noise = rng.normal(0.0, 1.0, size=(3, 6, NLAT, NLON)).astype(np.float32) * (noise_scale * std_arr)
        m_init = m_init + noise
        m_init[:, 1] = np.maximum(m_init[:, 1], 0.0)
        members.append(model_rollout(serve._MODEL, m_init, doy_sc, steps=steps, perturbation=None))
    return np.mean(np.stack(members, axis=0), axis=0)


def run_d(ds, splits) -> None:
    torch.set_num_threads(4)
    train_data = ds.data[splits.train]
    train_doy = ds.doy[splits.train]
    clim = fit_doy_climatology(train_data, train_doy)
    tp_idx = CHANNELS.index("tp")

    split_cache = {}
    for sname in ("val", "test"):
        s_slice = getattr(splits, sname)
        s_data = ds.data[s_slice]
        s_doy = ds.doy[s_slice]
        i_all, t_all, d_all, td_all = build_windows(s_data, s_doy, t_in=T_IN, horizon=8)
        idx = np.linspace(0, len(i_all) - 1, min(60, len(i_all)), dtype=int)
        inits, targs, doy_scs, tdoy = i_all[idx], t_all[idx], d_all[idx], td_all[idx]
        p_pers = predict_persistence(inits, horizon=8)
        p_clim = predict_climatology(clim, tdoy)
        best_rmse = {l: np.minimum(rmse_lat(p_pers[:, l-1], targs[:, l-1]), rmse_lat(p_clim[:, l-1], targs[:, l-1])) for l in LEAD_DAYS}
        best_csi = {l: float(np.nanmax([csi(p_pers[:, l-1, tp_idx], targs[:, l-1, tp_idx], RAIN_THRESH), csi(p_clim[:, l-1, tp_idx], targs[:, l-1, tp_idx], RAIN_THRESH)])) for l in LEAD_DAYS}
        split_cache[sname] = (inits, targs, doy_scs, best_rmse, best_csi)

    bob_box = (8.0, 20.0, 82.0, 94.0)
    bob_mask = box_mask(bob_box)
    s_inits, _, s_doy_scs, _, _ = split_cache["test"]
    sh_init = s_inits[min(30, len(s_inits) - 1)]
    sh_doy = s_doy_scs[min(30, len(s_doy_scs) - 1)]
    pert_3k = np.zeros((6, NLAT, NLON), dtype=np.float32)
    pert_3k[0, bob_mask] = 3.0
    pert_3k[5, bob_mask] = 0.15 * sh_init[-1, 5, bob_mask]

    print("\n=== TASK D: INFERENCE-ONLY ENSEMBLE SWEEP (NO RETRAINING) ===")
    for n_mem in (5, 8):
        for nscale in (0.015, 0.05, 0.1, 0.2):
            # Latency (10 steps, 10 calls, 4 cores)
            lats = []
            for _ in range(10):
                t0 = time.perf_counter()
                _ = _sim_custom_ensemble(sh_init, sh_doy, 10, None, nscale, n_mem)
                lats.append((time.perf_counter() - t0) * 1000.0)
            p95_ms = float(np.percentile(lats, 95))

            r_null = _sim_custom_ensemble(sh_init, sh_doy, 8, None, nscale, n_mem)
            r_zero = _sim_custom_ensemble(sh_init, sh_doy, 8, np.zeros((6, NLAT, NLON), dtype=np.float32), nscale, n_mem)
            zero_eq = bool(np.array_equal(r_null, r_zero))

            r_bob = _sim_custom_ensemble(sh_init, sh_doy, 8, pert_3k, nscale, n_mem)
            d_bob = r_bob - r_null
            in_anom = [round(float(d_bob[k, 0, bob_mask].mean()), 3) for k in (0, 1, 3, 7)]
            res_bob = causality_checks(r_null, r_bob, bob_box, STD_DEFAULT)

            print(f"\n[members={n_mem}, noise_scale={nscale}] p95_10step={p95_ms:.1f}ms | zero_eq={zero_eq} | BoB+3K near_d1={res_bob['near_frac_day1']:.4f} prop={res_bob['propagates']} in_t2m_anom(D1/2/4/8)={in_anom}")
            for sname in ("val", "test"):
                inits, targs, doy_scs, best_rmse, best_csi = split_cache[sname]
                preds = np.stack([_sim_custom_ensemble(inits[i], doy_scs[i], 8, None, nscale, n_mem) for i in range(len(inits))], axis=0)
                wins = []
                csis = []
                for l in LEAD_DAYS:
                    rm = rmse_lat(preds[:, l-1], targs[:, l-1])
                    wins.append(int((rm < best_rmse[l]).sum()))
                    csis.append(round(float(csi(preds[:, l-1, tp_idx], targs[:, l-1, tp_idx], RAIN_THRESH)), 4))
                vrs = np.nanmean(np.array([variance_ratio(preds[i, 7], targs[i, 7]) for i in range(len(inits))]), axis=0)
                vr_str = " ".join(f"{CHANNELS[c]}:{vrs[c]:.2f}" for c in range(6))
                print(f"  {sname:<4}: wins(D1/D4/D8)={wins[0]}/6,{wins[1]}/6,{wins[2]}/6 (tot={sum(wins)}/18) | CSI={csis} | D8_VR=[{vr_str}]")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    args = p.parse_args()
    ds = load_dataset(args.dataset)
    splits = compute_time_split(ds.data.shape[0])
    run_b(ds, splits)
    run_c(ds, splits)
    run_d(ds, splits)


if __name__ == "__main__":
    main()
