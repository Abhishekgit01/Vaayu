"""Judge-style shock evaluation on validation windows (no training, no model changes)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p != _SCRIPT_DIR]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.checkpoint import load_checkpoint
from model.net import MonsoonNet
from model.rollout import rollout
from monsoon_r1.contract import CHANNELS, NLAT, NLON, T_IN
from monsoon_r1.scoring import (
    BOX_ARABIAN_SEA,
    BOX_BAY_OF_BENGAL,
    LOCAL_MIN_FRAC,
    MAX_SPEED_CELLS_PER_DAY,
    NONZERO_MIN,
    NULL_TOL,
    STABLE_MAX_STD,
    STD_DEFAULT,
    box_mask,
    causality_checks,
    null_consistent,
    validity,
    variance_ratio,
)
from train.data import build_windows, compute_time_split, load_dataset


def _box_distance_grid(box: tuple[float, float, float, float]) -> np.ndarray:
    """Euclidean grid-cell distance from each (lat, lon) cell to the shock box."""
    ii, jj = np.indices((NLAT, NLON))
    bi, bj = np.where(box_mask(box))
    return np.sqrt(((ii[..., None] - bi) ** 2 + (jj[..., None] - bj) ** 2)).min(axis=-1)


def _front_speed_cells_per_day(
    diff: np.ndarray,
    std: np.ndarray,
    box: tuple[float, float, float, float],
) -> float:
    """Estimate outward front spread rate on days 4..8 in cells/day."""
    energy = ((diff / std[None, :, None, None]) ** 2).sum(axis=1)  # [steps, 32, 36]
    dist = _box_distance_grid(box)
    peak = float(energy.max())
    if peak <= 0.0:
        return 0.0

    thresh = 1e-4 * peak
    radii = []
    for k in range(energy.shape[0]):
        active = energy[k] >= thresh
        radii.append(float(dist[active].max()) if np.any(active) else 0.0)

    if len(radii) >= 8:
        # Spread from day 1 (idx 0) to days 4..8 (idx 3..7)
        late_radius = float(np.max(radii[3:8]))
        return max(0.0, (late_radius - radii[0]) / 4.0)
    if len(radii) >= 2:
        return max(0.0, (radii[-1] - radii[0]) / float(len(radii) - 1))
    return 0.0


def _build_shock_delta(name: str, last_state: np.ndarray) -> np.ndarray:
    """Build physical-unit shock delta [6, 32, 36] from init_states[-1]."""
    delta = np.zeros_like(last_state, dtype=np.float32)
    if name == "warm_ocean":
        mask = box_mask(BOX_BAY_OF_BENGAL)
        delta[0, mask] = 3.0
        delta[5, mask] = 0.15 * last_state[5, mask]
    elif name == "monsoon_stall":
        mask = box_mask(BOX_ARABIAN_SEA)
        delta[2, mask] = -0.40 * last_state[2, mask]
        delta[3, mask] = -0.40 * last_state[3, mask]
    elif name != "null":
        raise ValueError(f"Unknown shock name: {name}")
    return delta


def evaluate_window_shocks(
    model: MonsoonNet,
    init_state: np.ndarray,
    doy_sc: np.ndarray,
    target: np.ndarray,
    std: np.ndarray,
    steps: int = 8,
) -> dict[str, Any]:
    """Evaluate unshocked, warm_ocean, monsoon_stall, and null shocks on one window."""
    base = rollout(model, init_state, doy_sc, steps=steps, perturbation=None)
    base_valid = validity(base, steps)
    day8_idx = min(7, steps - 1)
    target_idx = min(day8_idx, target.shape[0] - 1)
    base_var_ratio = variance_ratio(base[day8_idx], target[target_idx])

    results: dict[str, Any] = {
        "base_valid": base_valid,
        "base_var_ratio": base_var_ratio,
        "shocks": {},
    }

    shock_specs = [
        ("warm_ocean", BOX_BAY_OF_BENGAL),
        ("monsoon_stall", BOX_ARABIAN_SEA),
    ]
    for name, box in shock_specs:
        delta = _build_shock_delta(name, init_state[-1])
        shocked = rollout(model, init_state, doy_sc, steps=steps, perturbation=delta)
        diff = shocked - base
        z = np.abs(diff / std[None, :, None, None])
        energy = float(((diff / std[None, :, None, None]) ** 2).sum())
        checks = causality_checks(base, shocked, box, std)
        spread_rate = _front_speed_cells_per_day(diff, std, box)

        results["shocks"][name] = {
            "max_abs": float(np.abs(diff).max()),
            "energy": energy,
            "max_std": float(z.max()),
            "near_frac_day1": min(1.0, max(0.0, float(checks.get("near_frac_day1", 0.0)))),
            "spread_cells_per_day": spread_rate,
            "far_frac_max": float(checks.get("far_frac_max", 0.0)),
            "nonzero": bool(checks["nonzero"]),
            "localized": bool(checks["localized"]),
            "propagates": bool(checks["propagates"]),
            "no_teleport": bool(checks.get("no_teleport", False)),
            "stable": bool(checks["stable"]),
            "valid": validity(shocked, steps),
            "day8_var_ratio": variance_ratio(shocked[day8_idx], target[target_idx]),
        }

    # Null shock (all zeros) vs unperturbed (None)
    zero_delta = _build_shock_delta("null", init_state[-1])
    null_out = rollout(model, init_state, doy_sc, steps=steps, perturbation=zero_delta)
    null_diff_std = float((np.abs(null_out - base) / std[None, :, None, None]).max())
    results["shocks"]["null"] = {
        "max_diff_std": null_diff_std,
        "consistent": bool(null_consistent(base, null_out, std)),
        "valid": validity(null_out, steps),
        "day8_var_ratio": variance_ratio(null_out[day8_idx], target[target_idx]),
    }
    return results


def evaluate_shocks(
    model: MonsoonNet,
    val_data: np.ndarray,
    val_doy: np.ndarray,
    std: np.ndarray,
    n_windows: int = 10,
    steps: int = 8,
) -> dict[str, Any]:
    """Run shock evaluation across n_windows evenly spaced validation windows."""
    horizon = max(8, steps)
    init_s, targets, doy_sc, _ = build_windows(val_data, val_doy, t_in=T_IN, horizon=horizon)
    if init_s.shape[0] < n_windows:
        raise ValueError(
            f"Requested {n_windows} validation windows, but only {init_s.shape[0]} available"
        )

    indices = np.linspace(0, init_s.shape[0] - 1, num=n_windows, dtype=int)
    per_window = [
        evaluate_window_shocks(model, init_s[i], doy_sc[i], targets[i], std, steps=steps)
        for i in indices
    ]

    summary: dict[str, Any] = {
        "n_windows": n_windows,
        "steps": steps,
        "window_indices": indices.tolist(),
        "per_window": per_window,
        "criteria": {},
    }

    for name in ("warm_ocean", "monsoon_stall"):
        rows = [w["shocks"][name] for w in per_window]
        mean_max_abs = float(np.mean([r["max_abs"] for r in rows]))
        mean_energy = float(np.mean([r["energy"] for r in rows]))
        mean_near = float(np.mean([r["near_frac_day1"] for r in rows]))
        mean_spread = float(np.mean([r["spread_cells_per_day"] for r in rows]))
        max_std_all = float(np.max([r["max_std"] for r in rows]))
        mean_var_ratio = np.mean([r["day8_var_ratio"] for r in rows], axis=0)
        all_valid = all(len(r["valid"]) == 0 for r in rows)

        summary["criteria"][name] = {
            "mean_max_abs": mean_max_abs,
            "mean_energy": mean_energy,
            "mean_near_frac_day1": mean_near,
            "mean_spread_cells_per_day": mean_spread,
            "max_std": max_std_all,
            "mean_day8_var_ratio": mean_var_ratio,
            "pass_nonzero": all(r["nonzero"] for r in rows) and mean_max_abs > 0.0,
            "pass_localized": mean_near >= LOCAL_MIN_FRAC,
            "pass_spread": mean_spread <= MAX_SPEED_CELLS_PER_DAY and all(r["no_teleport"] for r in rows),
            "pass_propagates": all(r["propagates"] for r in rows),
            "pass_stable": max_std_all <= STABLE_MAX_STD,
            "pass_valid": all_valid,
        }

    null_rows = [w["shocks"]["null"] for w in per_window]
    max_null_diff = float(np.max([r["max_diff_std"] for r in null_rows]))
    summary["criteria"]["null"] = {
        "max_diff_std": max_null_diff,
        "mean_diff_std": float(np.mean([r["max_diff_std"] for r in null_rows])),
        "pass_null": max_null_diff <= NULL_TOL and all(r["consistent"] for r in null_rows),
        "pass_valid": all(len(r["valid"]) == 0 for r in null_rows),
    }
    summary["base_all_valid"] = all(len(w["base_valid"]) == 0 for w in per_window)
    summary["base_mean_day8_var_ratio"] = np.mean(
        [w["base_var_ratio"] for w in per_window], axis=0
    )
    return summary


def print_report(summary: dict[str, Any]) -> None:
    """Print per-window and mean shock evaluation tables to stdout."""
    print(f"Shock Evaluation Summary (n_windows={summary['n_windows']}, steps={summary['steps']})")
    print(f"Base runs valid: {summary['base_all_valid']}")
    vr_str = ", ".join(f"{c}={v:.3f}" for c, v in zip(CHANNELS, summary["base_mean_day8_var_ratio"]))
    print(f"Base mean day-8 variance ratio: {vr_str}")

    for name in ("warm_ocean", "monsoon_stall"):
        print(f"\n=== Shock: {name} ===")
        print(
            f"{'win':>4} | {'max_abs':>9} | {'energy':>10} | {'near_d1':>8} | "
            f"{'spread/d':>8} | {'max_std':>8} | {'prop':>5} | {'valid':>5}"
        )
        for idx, w in enumerate(summary["per_window"]):
            r = w["shocks"][name]
            ok_v = "OK" if not r["valid"] else "FAIL"
            print(
                f"{idx:>4} | {r['max_abs']:>9.4f} | {r['energy']:>10.2f} | "
                f"{r['near_frac_day1']*100:>7.1f}% | {r['spread_cells_per_day']:>8.3f} | "
                f"{r['max_std']:>8.3f} | {str(r['propagates']):>5} | {ok_v:>5}"
            )
        c = summary["criteria"][name]
        print(
            f"MEAN | {c['mean_max_abs']:>9.4f} | {c['mean_energy']:>10.2f} | "
            f"{c['mean_near_frac_day1']*100:>7.1f}% | {c['mean_spread_cells_per_day']:>8.3f} | "
            f"{c['max_std']:>8.3f} (max)"
        )
        svr_str = ", ".join(f"{ch}={v:.3f}" for ch, v in zip(CHANNELS, c["mean_day8_var_ratio"]))
        print(f"Day-8 variance ratio: {svr_str}")
        print(
            "Criteria: "
            f"nonzero(>={NONZERO_MIN} std)={c['pass_nonzero']}, "
            f"localized(>={LOCAL_MIN_FRAC*100:.0f}%)={c['pass_localized']}, "
            f"spread(<={MAX_SPEED_CELLS_PER_DAY} cells/d)={c['pass_spread']}, "
            f"propagates={c['pass_propagates']}, "
            f"stable(<={STABLE_MAX_STD} std)={c['pass_stable']}, "
            f"valid={c['pass_valid']}"
        )

    print("\n=== Shock: null ===")
    print(f"{'win':>4} | {'max_diff_std':>12} | {'consistent':>10} | {'valid':>5}")
    for idx, w in enumerate(summary["per_window"]):
        r = w["shocks"]["null"]
        ok_v = "OK" if not r["valid"] else "FAIL"
        print(f"{idx:>4} | {r['max_diff_std']:>12.6e} | {str(r['consistent']):>10} | {ok_v:>5}")
    nc = summary["criteria"]["null"]
    print(
        f"Summary: mean={nc['mean_diff_std']:.6e}, max={nc['max_diff_std']:.6e}, "
        f"pass_null(<={NULL_TOL})={nc['pass_null']}, valid={nc['pass_valid']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge-style shock evaluation on validation windows.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint.")
    parser.add_argument("--dataset", required=True, help="Path to NetCDF dataset.")
    parser.add_argument("--n-windows", type=int, default=10, help="Number of validation windows (>= 10).")
    parser.add_argument("--steps", type=int, default=8, help="Rollout steps (>= 8).")
    args = parser.parse_args(argv)

    ds = load_dataset(args.dataset)
    split = compute_time_split(ds.data.shape[0])
    train_data = ds.data[split.train]
    std = np.std(train_data, axis=(0, 2, 3)).astype(np.float32)
    std = np.where(std > 1e-6, std, STD_DEFAULT).astype(np.float32)

    model = load_checkpoint(args.checkpoint, map_location="cpu")
    summary = evaluate_shocks(
        model,
        ds.data[split.val],
        ds.doy[split.val],
        std=std,
        n_windows=args.n_windows,
        steps=args.steps,
    )
    print_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
