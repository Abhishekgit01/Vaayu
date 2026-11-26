"""Seeded synthetic tests for train/shock_eval.py (no saved real data)."""
from __future__ import annotations

import numpy as np
import torch

from model.net import MonsoonNet
from monsoon_r1.contract import NLAT, NLON, T_IN
from monsoon_r1.scoring import STD_DEFAULT
from train.shock_eval import evaluate_shocks, print_report


def _make_tiny_model(seed: int = 0) -> MonsoonNet:
    torch.manual_seed(seed)
    mean = torch.tensor([295.0, 1.0, 0.0, 0.0, 101000.0, 35.0], dtype=torch.float32)
    std = torch.tensor([5.0, 1.0, 3.0, 3.0, 400.0, 10.0], dtype=torch.float32)
    static = torch.zeros((2, NLAT, NLON), dtype=torch.float32)
    model = MonsoonNet(static=static, mean=mean, std=std, hidden_dim=8)
    model.eval()
    return model


def _make_synthetic_val(n_days: int = 30, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = np.empty((n_days, 6, NLAT, NLON), dtype=np.float32)
    data[:, 0] = 296.0 + 3.0 * rng.standard_normal((n_days, NLAT, NLON))
    data[:, 1] = np.maximum(0.0, 2.0 + 3.0 * rng.standard_normal((n_days, NLAT, NLON)))
    data[:, 2] = 4.0 + 2.0 * rng.standard_normal((n_days, NLAT, NLON))
    data[:, 3] = 3.0 + 2.0 * rng.standard_normal((n_days, NLAT, NLON))
    data[:, 4] = 101000.0 + 300.0 * rng.standard_normal((n_days, NLAT, NLON))
    data[:, 5] = np.maximum(5.0, 35.0 + 6.0 * rng.standard_normal((n_days, NLAT, NLON)))
    doy = ((np.arange(n_days) % 365) + 1).astype(np.int32)
    return data, doy


def test_shock_eval_synthetic_end_to_end(capsys):
    model = _make_tiny_model(seed=1)
    val_data, val_doy = _make_synthetic_val(n_days=25, seed=7)
    assert val_data.shape[0] - T_IN - 8 + 1 >= 10

    summary = evaluate_shocks(
        model,
        val_data,
        val_doy,
        std=STD_DEFAULT,
        n_windows=10,
        steps=8,
    )
    assert summary["n_windows"] == 10
    assert summary["base_all_valid"] is True
    assert summary["criteria"]["null"]["pass_null"] is True
    assert summary["criteria"]["null"]["pass_valid"] is True

    for name in ("warm_ocean", "monsoon_stall"):
        crit = summary["criteria"][name]
        assert crit["mean_max_abs"] > 0.0
        assert crit["mean_energy"] > 0.0
        assert 0.0 <= crit["mean_near_frac_day1"] <= 1.0
        assert crit["mean_spread_cells_per_day"] >= 0.0
        assert crit["pass_valid"] is True

    print_report(summary)
    captured = capsys.readouterr().out
    assert "Shock: warm_ocean" in captured
    assert "Shock: monsoon_stall" in captured
    assert "Shock: null" in captured
