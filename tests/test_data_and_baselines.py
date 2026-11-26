"""Seeded synthetic tests for data loading, time splitting, and baseline evaluation."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import numpy as np
import xarray as xr

from monsoon_r1.contract import CHANNELS, LAT, LON, NLAT, NLON, T_IN
from train.baselines import fit_doy_climatology, predict_climatology, predict_persistence
from train.data import STATIC_CHANNELS, build_windows, compute_time_split, load_dataset
from train.eval_skill import evaluate_baselines


def _make_synthetic_nc(path: Path, n_days: int = 400, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0, scale=1.0, size=(n_days, len(CHANNELS), NLAT, NLON)).astype(np.float32)
    # Keep t2m in realistic Kelvin range and tp non-negative with occasional >10 mm/day
    data[:, 0] = 295.0 + 5.0 * data[:, 0]
    data[:, 1] = np.maximum(0.0, 4.0 + 6.0 * data[:, 1])
    static = rng.uniform(0.0, 1.0, size=(len(STATIC_CHANNELS), NLAT, NLON)).astype(np.float32)

    times = (np.datetime64(0, "D") + np.arange(n_days, dtype="timedelta64[D]")).astype("datetime64[ns]")
    ds = xr.Dataset(
        data_vars={
            "data": (("time", "channel", "lat", "lon"), data),
            "static": (("static_channel", "lat", "lon"), static),
        },
        coords={
            "time": times,
            "channel": list(CHANNELS),
            "lat": LAT,
            "lon": LON,
            "static_channel": list(STATIC_CHANNELS),
        },
    )
    ds.to_netcdf(path)
    return path


def test_time_split_fractions():
    split = compute_time_split(1000)
    assert split.train == slice(0, 800)
    assert split.val == slice(800, 900)
    assert split.test == slice(900, 1000)


def test_load_dataset_channel_order_and_static(tmp_path: Path):
    nc_path = _make_synthetic_nc(tmp_path / "synth.nc", n_days=120, seed=7)
    ds = load_dataset(nc_path)
    assert ds.data.shape == (120, 6, NLAT, NLON)
    assert ds.static.shape == (2, NLAT, NLON)
    assert ds.doy.shape == (120,)
    assert ds.lat[0] < ds.lat[-1]


def test_windows_and_baselines_in_memory():
    rng = np.random.default_rng(123)
    n_days = 100
    data = rng.normal(size=(n_days, 6, NLAT, NLON)).astype(np.float32)
    data[:, 1] = np.abs(data[:, 1]) * 8.0
    doy = ((np.arange(n_days) % 365) + 1).astype(np.int32)

    splits = compute_time_split(n_days)
    clim = fit_doy_climatology(data[splits.train], doy[splits.train])

    init_s, targets, doy_sc, target_doy = build_windows(
        data[splits.val], doy[splits.val], t_in=T_IN, horizon=4
    )
    assert init_s.shape[1:] == (T_IN, 6, NLAT, NLON)
    assert targets.shape[1:] == (4, 6, NLAT, NLON)
    assert doy_sc.shape == (init_s.shape[0], 2)

    pers = predict_persistence(init_s, horizon=4)
    clim_pred = predict_climatology(clim, target_doy)
    assert pers.shape == targets.shape
    assert clim_pred.shape == targets.shape
    assert np.array_equal(pers[:, 0], init_s[:, -1])


def test_eval_skill_end_to_end(tmp_path: Path):
    nc_path = _make_synthetic_nc(tmp_path / "synth_eval.nc", n_days=500, seed=99)
    summary = evaluate_baselines(nc_path, split_name="val")
    assert summary["n_windows"] > 0
    for lead in (1, 4, 8):
        r = summary["leads"][lead]
        assert r["rmse_persist"].shape == (6,)
        assert r["rmse_clim"].shape == (6,)
        assert np.all(np.isfinite(r["rmse_best"]))


def test_eval_skill_requires_dataset_arg():
    proc = subprocess.run(
        [sys.executable, "train/eval_skill.py"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
