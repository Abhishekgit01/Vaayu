"""Seeded synthetic tests for MonsoonNet, ChannelNormalizer, rollout, and checkpoints."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch

from model.checkpoint import load_checkpoint, save_checkpoint
from model.net import MonsoonNet, N_CHANNELS, NLAT, NLON, T_IN, count_parameters
from model.rollout import advance_doy_sincos, rollout


def _make_seeded_model_and_inputs(seed: int = 42):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    static = torch.from_numpy(rng.uniform(0.0, 1.0, size=(2, NLAT, NLON)).astype(np.float32))
    mean = torch.tensor([295.0, 1.2, 0.5, 0.2, 101000.0, 30.0], dtype=torch.float32)
    std = torch.tensor([6.0, 1.1, 3.0, 2.5, 450.0, 15.0], dtype=torch.float32)

    model = MonsoonNet(static=static, mean=mean, std=std, hidden_dim=64)
    model.eval()

    init_states = rng.normal(size=(T_IN, N_CHANNELS, NLAT, NLON)).astype(np.float32)
    init_states[:, 0] = 295.0 + 4.0 * init_states[:, 0]
    init_states[:, 1] = np.abs(init_states[:, 1]) * 5.0
    init_states[:, 4] = 101000.0 + 300.0 * init_states[:, 4]
    init_states[:, 5] = np.abs(init_states[:, 5]) * 20.0 + 10.0

    doy = 180
    doy_sc = [math.sin(2.0 * math.pi * doy / 365.0), math.cos(2.0 * math.pi * doy / 365.0)]
    return model, init_states, doy_sc, doy


def test_shapes_finiteness_and_rain_nonnegative():
    model, init_states, doy_sc, _ = _make_seeded_model_and_inputs(1)
    for steps in (10, 30):
        out = rollout(model, init_states, doy_sc, steps=steps, perturbation=None)
        assert out.shape == (steps, N_CHANNELS, NLAT, NLON)
        assert np.isfinite(out).all()
        assert out[:, 1].min() >= 0.0
        assert out[:, 0].min() >= 233.15
        assert out[:, 0].max() <= 333.15


def test_receptive_field_impulse_within_3_cells():
    model, init_states, doy_sc, _ = _make_seeded_model_and_inputs(2)
    ci, cj = 15, 17
    pert = np.zeros((N_CHANNELS, NLAT, NLON), dtype=np.float32)
    pert[0, ci, cj] = 2.0

    base = rollout(model, init_states, doy_sc, steps=1, perturbation=None)[0]
    imp = rollout(model, init_states, doy_sc, steps=1, perturbation=pert)[0]
    diff = np.abs(imp - base).max(axis=0)  # [32, 36]

    ii, jj = np.indices((NLAT, NLON))
    chebyshev_dist = np.maximum(np.abs(ii - ci), np.abs(jj - cj))
    outside_max = float(diff[chebyshev_dist > 3].max())
    inside_max = float(diff[chebyshev_dist <= 3].max())
    print(f"\n[receptive_field] inside_<=3_max={inside_max:.6e}, outside_>3_max={outside_max:.6e}")
    assert inside_max > 0.0
    assert outside_max == 0.0


def test_null_shock_exact_equality():
    model, init_states, doy_sc, _ = _make_seeded_model_and_inputs(3)
    zeros = np.zeros((N_CHANNELS, NLAT, NLON), dtype=np.float32)
    out_none = rollout(model, init_states, doy_sc, steps=10, perturbation=None)
    out_zero = rollout(model, init_states, doy_sc, steps=10, perturbation=zeros)
    assert np.array_equal(out_none, out_zero)


def test_determinism():
    model, init_states, doy_sc, _ = _make_seeded_model_and_inputs(4)
    out1 = rollout(model, init_states, doy_sc, steps=10, perturbation=None)
    out2 = rollout(model, init_states, doy_sc, steps=10, perturbation=None)
    assert np.array_equal(out1, out2)


def test_doy_rotation_accuracy():
    doy0 = 180
    sc0 = [math.sin(2.0 * math.pi * doy0 / 365.0), math.cos(2.0 * math.pi * doy0 / 365.0)]
    for k in (1, 10, 30):
        rot = advance_doy_sincos(sc0, steps=k)
        analytic = np.array(
            [
                math.sin(2.0 * math.pi * (doy0 + k) / 365.0),
                math.cos(2.0 * math.pi * (doy0 + k) / 365.0),
            ],
            dtype=np.float32,
        )
        assert np.allclose(rot, analytic, atol=1e-5)


def test_checkpoint_round_trip(tmp_path: Path):
    model, init_states, doy_sc, _ = _make_seeded_model_and_inputs(5)
    ckpt = tmp_path / "test_model.pt"
    save_checkpoint(ckpt, model, {"hidden_dim": model.hidden_dim})
    loaded = load_checkpoint(ckpt)

    out_orig = rollout(model, init_states, doy_sc, steps=10, perturbation=None)
    out_load = rollout(loaded, init_states, doy_sc, steps=10, perturbation=None)
    assert np.array_equal(out_orig, out_load)


def test_parameter_count_under_5m():
    model, _, _, _ = _make_seeded_model_and_inputs(6)
    n_params = count_parameters(model)
    print(f"\n[parameter_count] MonsoonNet params={n_params:,}")
    assert n_params < 5_000_000
