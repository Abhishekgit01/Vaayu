"""Evaluate persistence and training-split climatology baselines on the validation split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

# Allow running directly as `python train/eval_skill.py` as well as `python -m train.eval_skill`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from monsoon_r1.contract import CHANNELS, LEAD_DAYS, T_IN
from monsoon_r1.scoring import RAIN_THRESH, csi, rmse_lat
from train.baselines import fit_doy_climatology, predict_climatology, predict_persistence
from train.data import build_windows, compute_time_split, load_dataset


def evaluate_baselines(dataset_path: str | Path, split_name: str = "val") -> dict:
    """Compute latitude-weighted RMSE and heavy-rain CSI for persistence & climatology."""
    ds = load_dataset(dataset_path)
    splits = compute_time_split(ds.data.shape[0])
    split_slice = getattr(splits, split_name)

    train_data = ds.data[splits.train]
    train_doy = ds.doy[splits.train]
    clim = fit_doy_climatology(train_data, train_doy)

    horizon = max(LEAD_DAYS)
    eval_data = ds.data[split_slice]
    eval_doy = ds.doy[split_slice]
    init_states, targets, _, target_doy = build_windows(
        eval_data, eval_doy, t_in=T_IN, horizon=horizon
    )

    pred_persist = predict_persistence(init_states, horizon=horizon)
    pred_clim = predict_climatology(clim, target_doy)

    results = {}
    tp_idx = CHANNELS.index("tp")
    for lead in LEAD_DAYS:
        k = lead - 1
        y_true = targets[:, k]
        y_pers = pred_persist[:, k]
        y_clim = pred_clim[:, k]

        rmse_p = rmse_lat(y_pers, y_true)
        rmse_c = rmse_lat(y_clim, y_true)
        rmse_best = np.minimum(rmse_p, rmse_c)

        csi_p = csi(y_pers[:, tp_idx], y_true[:, tp_idx], thr=RAIN_THRESH)
        csi_c = csi(y_clim[:, tp_idx], y_true[:, tp_idx], thr=RAIN_THRESH)
        csi_best = float(np.nanmax([csi_p, csi_c])) if np.isfinite([csi_p, csi_c]).any() else float("nan")

        results[lead] = {
            "rmse_persist": rmse_p,
            "rmse_clim": rmse_c,
            "rmse_best": rmse_best,
            "csi_persist": csi_p,
            "csi_clim": csi_c,
            "csi_best": csi_best,
        }

    return {
        "n_time": int(ds.data.shape[0]),
        "split": split_name,
        "n_windows": int(init_states.shape[0]),
        "static_shape": tuple(ds.static.shape),
        "leads": results,
    }


def print_baseline_report(summary: dict) -> None:
    """Print per-channel latitude-weighted RMSE and heavy-rain CSI at leads 1, 4, 8."""
    print(
        f"Baseline evaluation | split={summary['split']} | "
        f"windows={summary['n_windows']} | total_days={summary['n_time']} | "
        f"static={summary['static_shape']}"
    )
    print("-" * 82)
    print(f"{'Lead':>4} | {'Channel':<6} | {'Persist RMSE':>13} | {'Clim RMSE':>13} | {'Better RMSE':>13} | {'Better':<7}")
    print("-" * 82)
    for lead in LEAD_DAYS:
        r = summary["leads"][lead]
        for ci, ch in enumerate(CHANNELS):
            rp = float(r["rmse_persist"][ci])
            rc = float(r["rmse_clim"][ci])
            rb = float(r["rmse_best"][ci])
            winner = "persist" if rp <= rc else "clim"
            print(f"{lead:>4} | {ch:<6} | {rp:>13.4f} | {rc:>13.4f} | {rb:>13.4f} | {winner:<7}")
        print("-" * 82)

    print("\nHeavy-rain CSI (tp > 10.0 mm/day):")
    print(f"{'Lead':>4} | {'Persist CSI':>12} | {'Clim CSI':>12} | {'Better CSI':>12}")
    print("-" * 48)
    for lead in LEAD_DAYS:
        r = summary["leads"][lead]
        print(
            f"{lead:>4} | {r['csi_persist']:>12.4f} | {r['csi_clim']:>12.4f} | {r['csi_best']:>12.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate persistence and climatology baselines on the validation split."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to india_climate_train.nc (required, no default).",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Evaluation split (default: val).",
    )
    args = parser.parse_args()
    summary = evaluate_baselines(args.dataset, split_name=args.split)
    print_baseline_report(summary)


if __name__ == "__main__":
    main()
