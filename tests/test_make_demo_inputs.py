"""Seeded synthetic tests for tools/make_demo_inputs.py."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import xarray as xr

from monsoon_r1.contract import CHANNELS, LAT, LON, NLAT, NLON, T_IN
from tools.make_demo_inputs import find_repo_root, select_window_index, write_scenario
from train.data import STATIC_CHANNELS


def _make_synthetic_nc(
    path: Path,
    n_days: int = 20,
    seed: int = 42,
    reversed_channels: bool = False,
) -> tuple[Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0, scale=1.0, size=(n_days, len(CHANNELS), NLAT, NLON)).astype(np.float32)
    data[:, 0] = 295.0 + 4.0 * data[:, 0]
    data[:, 1] = np.maximum(0.0, 3.0 + 5.0 * data[:, 1])
    data[:, 4] = 101000.0 + 300.0 * data[:, 4]
    data[:, 5] = np.maximum(5.0, 25.0 + 10.0 * data[:, 5])
    static = rng.uniform(0.0, 1.0, size=(len(STATIC_CHANNELS), NLAT, NLON)).astype(np.float32)

    times = (np.datetime64(0, "D") + np.arange(n_days, dtype="timedelta64[D]")).astype("datetime64[ns]")
    channel_names = list(CHANNELS)
    stored_data = data
    if reversed_channels:
        channel_names = list(reversed(CHANNELS))
        stored_data = data[:, ::-1, :, :]

    ds = xr.Dataset(
        data_vars={
            "data": (("time", "channel", "lat", "lon"), stored_data),
            "static": (("static_channel", "lat", "lon"), static),
        },
        coords={
            "time": times,
            "channel": channel_names,
            "lat": LAT,
            "lon": LON,
            "static_channel": list(STATIC_CHANNELS),
        },
    )
    ds.to_netcdf(path)
    doy = xr.DataArray(times, dims=("time",)).dt.dayofyear.values.astype(np.int32)
    return path, data, doy


def test_positions_and_contract_order_and_doy(tmp_path: Path):
    nc_path, expected_data, expected_doy = _make_synthetic_nc(
        tmp_path / "synth.nc", n_days=15, seed=11, reversed_channels=True
    )

    for pos, expected_start in ((0.0, 0), (0.5, 6), (1.0, 12)):
        out_path = tmp_path / f"scenario_{pos}.json"
        proc = subprocess.run(
            [
                sys.executable,
                "tools/make_demo_inputs.py",
                "--dataset",
                str(nc_path),
                "--out",
                str(out_path),
                "--position",
                str(pos),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert out_path.is_file()

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert set(payload.keys()) == {"init_states", "doy_sincos"}

        init_arr = np.asarray(payload["init_states"], dtype=np.float32)
        assert init_arr.shape == (T_IN, len(CHANNELS), NLAT, NLON)
        assert np.all(np.isfinite(init_arr))
        assert np.allclose(init_arr, expected_data[expected_start : expected_start + T_IN], atol=1e-5)

        last_doy = int(expected_doy[expected_start + T_IN - 1])
        expected_sc = [
            math.sin(2.0 * math.pi * last_doy / 365.0),
            math.cos(2.0 * math.pi * last_doy / 365.0),
        ]
        assert len(payload["doy_sincos"]) == 2
        assert all(math.isfinite(v) for v in payload["doy_sincos"])
        assert np.allclose(payload["doy_sincos"], expected_sc, atol=1e-6)

        # Never print data values
        combined_output = proc.stdout + proc.stderr
        assert "init_states" not in combined_output
        assert "doy_sincos" not in combined_output
        assert str(float(init_arr[0, 0, 0, 0])) not in combined_output


def test_refuses_to_write_inside_repo(tmp_path: Path):
    nc_path, _, _ = _make_synthetic_nc(tmp_path / "synth.nc", n_days=10, seed=5)
    repo_root = find_repo_root()
    bad_targets = [
        repo_root / "should_not_be_created.json",
        repo_root / "dashboard" / "should_not_be_created.json",
        Path("tools/should_not_be_created.json"),
    ]
    for bad_out in bad_targets:
        proc = subprocess.run(
            [
                sys.executable,
                "tools/make_demo_inputs.py",
                "--dataset",
                str(nc_path),
                "--out",
                str(bad_out),
                "--position",
                "0.5",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert not (repo_root / bad_out if not bad_out.is_absolute() else bad_out).exists()


def test_requires_all_cli_args_no_defaults(tmp_path: Path):
    nc_path, _, _ = _make_synthetic_nc(tmp_path / "synth.nc", n_days=10, seed=6)
    out_path = tmp_path / "out.json"
    bad_arg_lists = [
        [],
        ["--dataset", str(nc_path), "--out", str(out_path)],
        ["--dataset", str(nc_path), "--position", "0.5"],
        ["--out", str(out_path), "--position", "0.5"],
    ]
    for argv in bad_arg_lists:
        proc = subprocess.run(
            [sys.executable, "tools/make_demo_inputs.py", *argv],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0


@pytest.mark.parametrize("bad_pos", [-0.01, 1.01, float("nan"), float("inf")])
def test_rejects_out_of_range_position(tmp_path: Path, bad_pos: float):
    nc_path, _, _ = _make_synthetic_nc(tmp_path / "synth.nc", n_days=10, seed=7)
    out_path = tmp_path / "bad_pos.json"
    with pytest.raises(ValueError):
        write_scenario(nc_path, out_path, bad_pos)
    assert not out_path.exists()


def test_rejects_non_finite_window(tmp_path: Path):
    nc_path, data, _ = _make_synthetic_nc(tmp_path / "synth_nan.nc", n_days=10, seed=8)
    with xr.open_dataset(nc_path) as ds:
        loaded = ds.load()
    loaded["data"].values[1, 0, 0, 0] = np.nan
    nan_path = tmp_path / "nan.nc"
    loaded.to_netcdf(nan_path)

    out_path = tmp_path / "nan_out.json"
    with pytest.raises(ValueError, match="non-finite"):
        write_scenario(nan_path, out_path, 0.0)
    assert not out_path.exists()


def test_select_window_index_bounds():
    assert select_window_index(3, 0.0) == 0
    assert select_window_index(3, 1.0) == 0
    assert select_window_index(10, 0.0) == 0
    assert select_window_index(10, 1.0) == 7
