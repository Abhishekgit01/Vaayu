"""Seeded synthetic tests for shock fine-tuning losses, synthetic shock sampler, and smoke run."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import xarray as xr

from model.checkpoint import load_checkpoint, save_checkpoint
from model.net import MonsoonNet, N_CHANNELS, NLAT, NLON
from monsoon_r1.contract import CHANNELS, LAT, LON, lat_weights
from train.data import STATIC_CHANNELS
from train.losses import channel_weighted_forecast_loss, shock_causality_loss
from train.train_shock import sample_synthetic_shocks


def _make_synthetic_nc(path: Path, n_days: int = 120, seed: int = 21) -> Path:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_days, len(CHANNELS), NLAT, NLON)).astype(np.float32)
    data[:, 0] = 295.0 + 4.0 * data[:, 0]
    data[:, 1] = np.maximum(0.0, 3.0 + 6.0 * data[:, 1])
    data[:, 4] = 101000.0 + 300.0 * data[:, 4]
    data[:, 5] = np.maximum(5.0, 30.0 + 10.0 * data[:, 5])
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


def test_sample_synthetic_shocks_bounded_and_localized():
    rng = np.random.default_rng(42)
    last = torch.full((4, N_CHANNELS, NLAT, NLON), 20.0, dtype=torch.float32)
    last[:, 0] = 295.0
    pert, near_m, dist_g = sample_synthetic_shocks(last, rng)
    assert pert.shape == (4, N_CHANNELS, NLAT, NLON)
    assert near_m.shape == (4, NLAT, NLON)
    assert dist_g.shape == (4, NLAT, NLON)
    assert torch.isfinite(pert).all()
    assert torch.abs(pert[:, 0]).max().item() <= 4.0 + 1e-5
    assert near_m.min().item() >= 0.0 and near_m.max().item() <= 1.0


def test_shock_causality_loss_grads_and_channel_weighted_loss():
    torch.manual_seed(5)
    base = torch.randn(2, 8, N_CHANNELS, NLAT, NLON)
    shocked = (base + 0.1 * torch.randn_like(base)).detach().requires_grad_(True)
    near_m = torch.zeros(2, NLAT, NLON)
    near_m[:, 10:18, 12:20] = 1.0
    dist_g = torch.zeros(2, NLAT, NLON)
    std_p = torch.tensor([8.0, 8.0, 3.5, 2.5, 450.0, 20.0], dtype=torch.float32)

    loss_sh, terms_sh = shock_causality_loss(base, shocked, near_m, dist_g, std_p)
    assert torch.isfinite(loss_sh)
    loss_sh.backward()
    assert shocked.grad is not None and shocked.grad.abs().max().item() > 0.0
    assert "shock_prop" in terms_sh

    pred = torch.randn(2, 4, N_CHANNELS, NLAT, NLON, requires_grad=True)
    target = torch.randn(2, 4, N_CHANNELS, NLAT, NLON)
    tp_phys = torch.rand(2, 4, NLAT, NLON) * 15.0
    w_lat = torch.tensor(lat_weights(), dtype=torch.float32)
    cw = torch.tensor([2.5, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    loss_fc, terms_fc = channel_weighted_forecast_loss(pred, target, tp_phys, w_lat, cw)
    assert torch.isfinite(loss_fc)
    assert "tcwv_cons" in terms_fc
    loss_fc.backward()
    assert pred.grad is not None and pred.grad.abs().max().item() > 0.0


def test_smoke_train_shock_run(tmp_path: Path):
    nc_path = _make_synthetic_nc(tmp_path / "synth.nc", n_days=120, seed=31)
    init_ckpt = tmp_path / "init.pt"
    out_ckpt = tmp_path / "tuned.pt"

    m = MonsoonNet(hidden_dim=16)
    save_checkpoint(init_ckpt, m, {"hidden_dim": 16})

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "train.train_shock",
            "--dataset",
            str(nc_path),
            "--checkpoint",
            str(init_ckpt),
            "--out",
            str(out_ckpt),
            "--epochs",
            "1",
            "--horizon",
            "8",
            "--batch-size",
            "4",
            "--shock-prob",
            "1.0",
            "--smoke",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert out_ckpt.is_file()
    loaded = load_checkpoint(out_ckpt)
    assert loaded.hidden_dim == 16
