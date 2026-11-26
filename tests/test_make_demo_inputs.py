"""Synthetic tests for tools/make_demo_inputs.py."""
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
from tools.make_demo_inputs import REPO_ROOT, build_scenario, ensure_outside_repo, write_scenario

STATIC_CHANNELS = ("land_sea_mask", "orography_m")


def _make_synthetic_nc(
    path: Path,
    n_days: int = 30,
    seed: int = 123,
    inject_nan: bool = False,
) -> tuple[Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0, scale=1.0, size=(n_days, len(CHANNELS), NLAT, NLON)).astype(np.float32)
    data[:, 0] = 295.0 + 4.0 * data[:, 0]
    data[:, 1] = np.maximum(0.0, 3.0 + 5.0 * data[:, 1])
    if inject_nan:
        data[5, 0, 0, 0] = np.nan

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
    doy = np.asarray(ds["time"].dt.dayofyear.values, dtype=np.int32)
    return path, data, doy


def test_refuses_output_inside_git_repo(tmp_path: Path):
    nc_path, _, _ = _make_synthetic_nc(tmp_path / "synth.nc", n_days=10, seed=1)
    inside_repo = REPO_ROOT / "tools" / "should_not_exist.json"

    with pytest.raises(ValueError, match="git repository"):
        ensure_outside_repo(inside_repo)

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "make_demo_inputs.py"),
            "--dataset",
            str(nc_path),
            "--out",
            str(inside_repo),
            "--position",
            "0.5",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert not inside_repo.exists()


@pytest.mark.parametrize("position,expected_start", [(0.0, 0), (0.5, 9), (1.0, 18)])
def test_generates_valid_scenario_at_positions(tmp_path: Path, position: float, expected_start: int):
    nc_path, data, doy = _make_synthetic_nc(tmp_path / "synth.nc", n_days=21, seed=7)
    out_file = tmp_path / "external" / f"scenario_{position}.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "make_demo_inputs.py"),
            "--dataset",
            str(nc_path),
            "--out",
            str(out_file),
            "--position",
            str(position),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_file.is_file()

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    init_states = np.asarray(payload["init_states"], dtype=np.float32)
    doy_sincos = payload["doy_sincos"]

    assert init_states.shape == (T_IN, len(CHANNELS), NLAT, NLON)
    assert np.all(np.isfinite(init_states))
    np.testing.assert_allclose(init_states, data[expected_start : expected_start + T_IN], rtol=1e-6, atol=1e-6)

    last_doy = float(doy[expected_start + T_IN - 1])
    expected_sincos = [
        math.sin(2.0 * math.pi * last_doy / 365.0),
        math.cos(2.0 * math.pi * last_doy / 365.0),
    ]
    assert len(doy_sincos) == 2
    assert all(math.isfinite(v) for v in doy_sincos)
    assert doy_sincos == pytest.approx(expected_sincos, abs=1e-6)


def test_rejects_invalid_position_and_nonfinite_data(tmp_path: Path):
    nc_path, _, _ = _make_synthetic_nc(tmp_path / "synth_nan.nc", n_days=10, seed=3, inject_nan=True)

    with pytest.raises(ValueError, match="position"):
        build_scenario(nc_path, -0.1)
    with pytest.raises(ValueError, match="position"):
        build_scenario(nc_path, 1.5)
    with pytest.raises(ValueError, match="Non-finite"):
        write_scenario(nc_path, tmp_path / "out.json", 0.5)
    assert not (tmp_path / "out.json").exists()
