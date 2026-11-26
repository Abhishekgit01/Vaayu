"""Seeded synthetic tests for losses, differentiable rollout parity, and smoke training."""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import xarray as xr

from model.checkpoint import load_checkpoint
from model.net import MonsoonNet, N_CHANNELS, NLAT, NLON, T_IN
from model.rollout import rollout
from monsoon_r1.contract import CHANNELS, LAT, LON
from train.data import STATIC_CHANNELS
from train.evaluate import rollout_differentiable
from train.losses import MonsoonLoss


def _make_synthetic_nc(path: Path, n_days: int = 120, seed: int = 11) -> Path:
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


def test_loss_terms_finite_and_nonzero_grads():
    torch.manual_seed(7)
    pred = torch.randn(2, 4, N_CHANNELS, NLAT, NLON, requires_grad=True)
    target = torch.randn(2, 4, N_CHANNELS, NLAT, NLON)
    tp_phys = torch.rand(2, 4, NLAT, NLON) * 20.0

    criterion = MonsoonLoss(w_mse=1.0, w_rain_extra=4.0, w_grad=0.2, w_var=0.1)
    loss, terms = criterion(pred, target, tp_phys)
    assert torch.isfinite(loss)
    for k, v in terms.items():
        assert math.isfinite(v), f"{k} not finite"
        assert v > 0.0, f"{k} expected > 0"
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().max().item() > 0.0


def test_variance_term_zero_when_variances_match():
    torch.manual_seed(8)
    target = torch.randn(2, 3, N_CHANNELS, NLAT, NLON)
    # Shifted copy has identical spatial variance but different mean
    pred = target + 2.5
    tp_phys = torch.zeros(2, 3, NLAT, NLON)

    criterion = MonsoonLoss(w_mse=0.0, w_rain_extra=0.0, w_grad=0.0, w_var=0.1)
    loss, terms = criterion(pred, target, tp_phys)
    assert abs(terms["var_match"]) < 1e-6
    assert abs(float(loss.item())) < 1e-6


def test_differentiable_rollout_parity_with_model_rollout():
    torch.manual_seed(9)
    rng = np.random.default_rng(9)
    static = torch.from_numpy(rng.uniform(0.0, 1.0, size=(2, NLAT, NLON)).astype(np.float32))
    mean = torch.tensor([295.0, 1.2, 0.5, 0.2, 101000.0, 30.0], dtype=torch.float32)
    std = torch.tensor([6.0, 1.1, 3.0, 2.5, 450.0, 15.0], dtype=torch.float32)
    model = MonsoonNet(static=static, mean=mean, std=std, hidden_dim=32)
    model.eval()

    init_np = rng.normal(size=(T_IN, N_CHANNELS, NLAT, NLON)).astype(np.float32)
    init_np[:, 0] = 295.0 + 3.0 * init_np[:, 0]
    init_np[:, 1] = np.abs(init_np[:, 1]) * 4.0
    init_np[:, 4] = 101000.0 + 300.0 * init_np[:, 4]
    init_np[:, 5] = np.abs(init_np[:, 5]) * 15.0 + 10.0
    doy_sc = np.array([math.sin(1.0), math.cos(1.0)], dtype=np.float32)

    ref = rollout(model, init_np, doy_sc, steps=8, perturbation=None)
    with torch.no_grad():
        _, bat = rollout_differentiable(
            model,
            torch.from_numpy(init_np).unsqueeze(0),
            torch.from_numpy(doy_sc).unsqueeze(0),
            steps=8,
            noise_std=0.0,
        )
    diff = np.abs(ref - bat.squeeze(0).numpy()).max()
    assert diff < 1e-4, f"Parity max diff {diff} >= 1e-4"


def test_smoke_training_run_creates_checkpoint(tmp_path: Path):
    nc_path = _make_synthetic_nc(tmp_path / "synth_train.nc", n_days=140, seed=15)
    ckpt_path = tmp_path / "smoke.pt"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "train.train",
            "--dataset",
            str(nc_path),
            "--out",
            str(ckpt_path),
            "--epochs",
            "2",
            "--width",
            "16",
            "--batch-size",
            "8",
            "--smoke",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert ckpt_path.is_file()
    loaded = load_checkpoint(ckpt_path)
    assert loaded.hidden_dim == 16
