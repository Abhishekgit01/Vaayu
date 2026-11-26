"""Contract tests for serve.py using only stdlib, pytest, and numpy."""
from __future__ import annotations

import json
import math
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NLAT = 32
NLON = 36
N_CHANNELS = 6
T_IN = 3


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _post_json(url: str, payload: dict, timeout: float = 15.0) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return resp.status, body


def _make_synthetic_init_states(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = np.empty((T_IN, N_CHANNELS, NLAT, NLON), dtype=np.float32)
    states[:, 0] = 296.0 + 4.0 * rng.standard_normal((T_IN, NLAT, NLON))           # t2m (K)
    states[:, 1] = np.maximum(0.0, 4.0 + 5.0 * rng.standard_normal((T_IN, NLAT, NLON)))  # tp (mm/day)
    states[:, 2] = 3.0 * rng.standard_normal((T_IN, NLAT, NLON))                   # u10 (m/s)
    states[:, 3] = 3.0 * rng.standard_normal((T_IN, NLAT, NLON))                   # v10 (m/s)
    states[:, 4] = 101000.0 + 400.0 * rng.standard_normal((T_IN, NLAT, NLON))      # msl (Pa)
    states[:, 5] = np.maximum(5.0, 35.0 + 8.0 * rng.standard_normal((T_IN, NLAT, NLON))) # tcwv (kg/m2)
    return states


def _doy_sincos(doy: int = 180) -> list[float]:
    angle = 2.0 * math.pi * float(doy) / 365.0
    return [math.sin(angle), math.cos(angle)]


def _assert_valid_rollout(rollout: np.ndarray, expected_steps: int) -> None:
    assert rollout.shape == (expected_steps, N_CHANNELS, NLAT, NLON)
    assert np.all(np.isfinite(rollout)), "Rollout contains NaN or Inf"
    assert np.all(rollout[:, 1] >= 0.0), "Precipitation tp must be >= 0"
    assert np.all((rollout[:, 0] >= 233.15) & (rollout[:, 0] <= 333.15)), (
        "Temperature t2m out of [233.15, 333.15] K bounds"
    )


@pytest.fixture(scope="module")
def server_url():
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "serve.py"), "--port", str(port)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    deadline = time.monotonic() + 30.0
    ready = False
    last_err = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out, err = proc.communicate()
            pytest.fail(f"serve.py exited early (code={proc.returncode}):\n{out}\n{err}")
        try:
            status, body = _post_json(f"{base_url}/health", {}, timeout=2.0)
            if status == 200 and body.get("status") == "ready":
                ready = True
                break
        except Exception as exc:
            last_err = exc
            time.sleep(0.25)

    if not ready:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.fail(f"Server did not become ready within 30s (last error: {last_err})")

    try:
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_health_ready(server_url: str):
    status, body = _post_json(f"{server_url}/health", {})
    assert status == 200
    assert body.get("status") == "ready"
    assert isinstance(body.get("model_name"), str) and len(body["model_name"]) > 0


def test_simulate_10_steps_contract_and_speed(server_url: str):
    init_states = _make_synthetic_init_states(seed=11)
    payload = {
        "init_states": init_states.tolist(),
        "doy_sincos": _doy_sincos(172),
        "steps": 10,
        "perturbation": None,
    }

    t0 = time.perf_counter()
    status, body = _post_json(f"{server_url}/simulate", payload, timeout=10.0)
    elapsed = time.perf_counter() - t0

    assert status == 200
    assert elapsed < 5.0, f"10-step /simulate took {elapsed:.3f}s (limit 5.0s)"
    assert isinstance(body.get("inference_ms"), (int, float))
    assert math.isfinite(float(body["inference_ms"]))

    rollout = np.asarray(body["rollout"], dtype=np.float32)
    _assert_valid_rollout(rollout, expected_steps=10)


def test_null_vs_zero_perturbation_equivalence(server_url: str):
    init_states = _make_synthetic_init_states(seed=22)
    doy_sc = _doy_sincos(200)

    _, body_none = _post_json(
        f"{server_url}/simulate",
        {
            "init_states": init_states.tolist(),
            "doy_sincos": doy_sc,
            "steps": 10,
            "perturbation": None,
        },
    )
    _, body_zero = _post_json(
        f"{server_url}/simulate",
        {
            "init_states": init_states.tolist(),
            "doy_sincos": doy_sc,
            "steps": 10,
            "perturbation": np.zeros((N_CHANNELS, NLAT, NLON), dtype=np.float32).tolist(),
        },
    )

    rollout_none = np.asarray(body_none["rollout"], dtype=np.float32)
    rollout_zero = np.asarray(body_zero["rollout"], dtype=np.float32)
    _assert_valid_rollout(rollout_none, 10)
    _assert_valid_rollout(rollout_zero, 10)

    channel_std = np.maximum(np.std(rollout_none, axis=(0, 2, 3)), 1e-6)
    max_diff = np.max(np.abs(rollout_zero - rollout_none), axis=(0, 2, 3))
    assert np.all(max_diff <= 1e-3 * channel_std), (
        f"Zero perturbation diverged from null: max_diff={max_diff}, tol={1e-3 * channel_std}"
    )


@pytest.mark.parametrize("steps", [1, 30])
def test_steps_1_and_30_work(server_url: str, steps: int):
    init_states = _make_synthetic_init_states(seed=33 + steps)
    status, body = _post_json(
        f"{server_url}/simulate",
        {
            "init_states": init_states.tolist(),
            "doy_sincos": _doy_sincos(90),
            "steps": steps,
            "perturbation": None,
        },
        timeout=20.0,
    )
    assert status == 200
    rollout = np.asarray(body["rollout"], dtype=np.float32)
    _assert_valid_rollout(rollout, expected_steps=steps)


def test_wrong_shapes_return_error_and_server_survives(server_url: str):
    init_states = _make_synthetic_init_states(seed=44)
    bad_payloads = [
        {
            "init_states": init_states[:2].tolist(),  # wrong time length: 2 instead of 3
            "doy_sincos": _doy_sincos(150),
            "steps": 5,
            "perturbation": None,
        },
        {
            "init_states": init_states.tolist(),
            "doy_sincos": _doy_sincos(150),
            "steps": 5,
            "perturbation": np.zeros((N_CHANNELS, 10, 10), dtype=np.float32).tolist(),  # wrong spatial shape
        },
    ]

    for bad in bad_payloads:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_json(f"{server_url}/simulate", bad)
        assert exc_info.value.code >= 400

    # Server must still be healthy and able to serve valid requests
    status, body = _post_json(f"{server_url}/health", {})
    assert status == 200
    assert body.get("status") == "ready"

    status, body = _post_json(
        f"{server_url}/simulate",
        {
            "init_states": init_states.tolist(),
            "doy_sincos": _doy_sincos(150),
            "steps": 2,
            "perturbation": None,
        },
    )
    assert status == 200
    _assert_valid_rollout(np.asarray(body["rollout"], dtype=np.float32), expected_steps=2)


def _post_raw(url: str, raw: bytes, timeout: float = 10.0) -> int:
    req = urllib.request.Request(
        url,
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_robustness_bad_inputs_and_recovery(server_url: str):
    init_states = _make_synthetic_init_states(seed=55)
    doy_sc = _doy_sincos(170)

    # 1. Empty body -> 400
    assert _post_raw(f"{server_url}/simulate", b"") == 400

    # 2. Bad JSON -> 400
    assert _post_raw(f"{server_url}/simulate", b"{not valid json") == 400

    # 3. NaN in payload -> 422
    nan_body = json.dumps({
        "init_states": init_states.tolist(),
        "doy_sincos": [0.5, 0.5],
        "steps": 2,
        "perturbation": None,
    }).replace("[0.5, 0.5]", "[NaN, 0.5]", 1).encode("utf-8")
    assert _post_raw(f"{server_url}/simulate", nan_body) == 422

    # 4. Wrong shape -> 422
    wrong_shape = json.dumps({
        "init_states": init_states[:2].tolist(),
        "doy_sincos": doy_sc,
        "steps": 2,
        "perturbation": None,
    }).encode("utf-8")
    assert _post_raw(f"{server_url}/simulate", wrong_shape) == 422

    # 5. Out-of-range physical state (e.g. t2m = 999 K) -> 422
    bad_state = init_states.copy()
    bad_state[0, 0, 0, 0] = 999.0
    assert _post_raw(
        f"{server_url}/simulate",
        json.dumps({"init_states": bad_state.tolist(), "doy_sincos": doy_sc, "steps": 2, "perturbation": None}).encode("utf-8"),
    ) == 422

    # 6. Huge shock (+1e4 K) -> 422
    huge_pert = np.zeros((N_CHANNELS, NLAT, NLON), dtype=np.float32)
    huge_pert[0, 0, 0] = 1e4
    assert _post_raw(
        f"{server_url}/simulate",
        json.dumps({"init_states": init_states.tolist(), "doy_sincos": doy_sc, "steps": 2, "perturbation": huge_pert.tolist()}).encode("utf-8"),
    ) == 422

    # 7. Valid request immediately afterwards still succeeds, plus determinism & byte-identical zero shock
    status1, body1 = _post_json(
        f"{server_url}/simulate",
        {"init_states": init_states.tolist(), "doy_sincos": doy_sc, "steps": 5, "perturbation": None},
    )
    status2, body2 = _post_json(
        f"{server_url}/simulate",
        {"init_states": init_states.tolist(), "doy_sincos": doy_sc, "steps": 5, "perturbation": None},
    )
    status_z, body_z = _post_json(
        f"{server_url}/simulate",
        {
            "init_states": init_states.tolist(),
            "doy_sincos": doy_sc,
            "steps": 5,
            "perturbation": np.zeros((N_CHANNELS, NLAT, NLON), dtype=np.float32).tolist(),
        },
    )
    assert status1 == 200 and status2 == 200 and status_z == 200
    r1 = np.asarray(body1["rollout"], dtype=np.float32)
    r2 = np.asarray(body2["rollout"], dtype=np.float32)
    rz = np.asarray(body_z["rollout"], dtype=np.float32)
    _assert_valid_rollout(r1, 5)
    assert np.array_equal(r1, r2)
    assert np.array_equal(r1, rz)


def test_state_bounds_headroom(server_url: str) -> None:
    """Extreme-but-valid t2m=355 K and msl=120000 Pa return 200; out-of-bounds return 422."""
    rng = np.random.default_rng(7)
    init_states = np.zeros((3, N_CHANNELS, NLAT, NLON), dtype=np.float32)
    init_states[:, 0] = 300.0 + rng.normal(0.0, 2.0, (3, NLAT, NLON))
    init_states[:, 1] = np.clip(rng.gamma(1.5, 4.0, (3, NLAT, NLON)), 0.0, 200.0)
    init_states[:, 2] = rng.normal(2.0, 3.0, (3, NLAT, NLON))
    init_states[:, 3] = rng.normal(1.0, 3.0, (3, NLAT, NLON))
    init_states[:, 4] = 100800.0 + rng.normal(0.0, 200.0, (3, NLAT, NLON))
    init_states[:, 5] = 45.0 + rng.normal(0.0, 5.0, (3, NLAT, NLON))
    doy_sc = [0.0, -1.0]

    # Extreme-but-valid values within widened STATE_BOUNDS
    valid_ext = init_states.copy()
    valid_ext[0, 0, 0, 0] = 355.0
    valid_ext[0, 4, 0, 0] = 120000.0
    status_ok, body_ok = _post_json(
        f"{server_url}/simulate",
        {"init_states": valid_ext.tolist(), "doy_sincos": doy_sc, "steps": 2, "perturbation": None},
    )
    assert status_ok == 200
    _assert_valid_rollout(np.asarray(body_ok["rollout"], dtype=np.float32), 2)

    # Out-of-bounds msl = 50000 Pa -> 422
    bad_msl = init_states.copy()
    bad_msl[0, 4, 0, 0] = 50000.0
    assert _post_raw(
        f"{server_url}/simulate",
        json.dumps({"init_states": bad_msl.tolist(), "doy_sincos": doy_sc, "steps": 2, "perturbation": None}).encode("utf-8"),
    ) == 422
