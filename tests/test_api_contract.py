"""End-to-end API contract tests against `python serve.py --port <free_port>`."""
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _post_json(url: str, payload: dict, timeout: float = 30.0) -> tuple[int, dict | str]:
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: dict | str = json.loads(raw)
        except Exception:
            parsed = raw
        return int(exc.code), parsed


def _make_synthetic_inputs(seed: int = 42) -> tuple[np.ndarray, list[float], np.ndarray]:
    rng = np.random.default_rng(seed)
    init_states = np.empty((3, 6, 32, 36), dtype=np.float32)
    init_states[:, 0] = np.clip(295.0 + 4.0 * rng.normal(size=(3, 32, 36)), 240.0, 325.0)
    init_states[:, 1] = np.maximum(0.0, 3.0 + 5.0 * rng.normal(size=(3, 32, 36)))
    init_states[:, 2] = rng.normal(loc=0.5, scale=3.0, size=(3, 32, 36))
    init_states[:, 3] = rng.normal(loc=0.2, scale=2.5, size=(3, 32, 36))
    init_states[:, 4] = 101000.0 + 300.0 * rng.normal(size=(3, 32, 36))
    init_states[:, 5] = np.maximum(1.0, 30.0 + 10.0 * rng.normal(size=(3, 32, 36)))

    doy = 180
    angle = 2.0 * math.pi * float(doy) / 365.0
    doy_sincos = [float(math.sin(angle)), float(math.cos(angle))]
    channel_std = np.std(init_states, axis=(0, 2, 3))
    return init_states, doy_sincos, channel_std


@pytest.fixture(scope="module")
def server_url():
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [sys.executable, str(_REPO_ROOT / "serve.py"), "--port", str(port)],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 120.0
        last_err: str | None = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=5)
                pytest.fail(
                    f"Server exited prematurely with code {proc.returncode}.\n"
                    f"stdout:\n{out}\nstderr:\n{err}"
                )
            try:
                status, data = _post_json(f"{base_url}/health", {}, timeout=2.0)
                if status == 200 and isinstance(data, dict) and data.get("status") == "ready":
                    break
                last_err = f"status={status}, body={data}"
            except Exception as exc:
                last_err = repr(exc)
            time.sleep(0.25)
        else:
            pytest.fail(f"Server did not become ready within 120 s (last error: {last_err})")

        yield base_url
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()


def _assert_valid_rollout(arr: np.ndarray, expected_steps: int) -> None:
    assert arr.shape == (expected_steps, 6, 32, 36)
    assert np.all(np.isfinite(arr))
    assert float(arr[:, 1].min()) >= 0.0
    assert float(arr[:, 0].min()) >= 233.15
    assert float(arr[:, 0].max()) <= 333.15


def test_health_ready(server_url: str):
    status, data = _post_json(f"{server_url}/health", {})
    assert status == 200
    assert isinstance(data, dict)
    assert data.get("status") == "ready"
    assert isinstance(data.get("model_name"), str) and len(data["model_name"]) > 0


def test_simulate_10_steps_physical_and_wall_time(server_url: str):
    init_states, doy_sincos, _ = _make_synthetic_inputs(seed=10)
    payload = {
        "init_states": init_states.tolist(),
        "doy_sincos": doy_sincos,
        "steps": 10,
        "perturbation": None,
    }
    t0 = time.perf_counter()
    status, data = _post_json(f"{server_url}/simulate", payload)
    wall_s = time.perf_counter() - t0
    print(f"\n[api_contract] 10-step wall time: {wall_s:.4f} s")

    assert status == 200
    assert isinstance(data, dict)
    assert wall_s < 5.0
    assert isinstance(data.get("inference_ms"), (int, float))
    assert math.isfinite(float(data["inference_ms"]))

    rollout = np.asarray(data["rollout"], dtype=np.float32)
    _assert_valid_rollout(rollout, expected_steps=10)


def test_null_vs_zero_perturbation_equivalence(server_url: str):
    init_states, doy_sincos, channel_std = _make_synthetic_inputs(seed=20)
    assert np.all(channel_std > 0.0)

    base_payload = {
        "init_states": init_states.tolist(),
        "doy_sincos": doy_sincos,
        "steps": 10,
        "perturbation": None,
    }
    zero_payload = {
        "init_states": init_states.tolist(),
        "doy_sincos": doy_sincos,
        "steps": 10,
        "perturbation": np.zeros((6, 32, 36), dtype=np.float32).tolist(),
    }

    s_null, d_null = _post_json(f"{server_url}/simulate", base_payload)
    s_zero, d_zero = _post_json(f"{server_url}/simulate", zero_payload)
    assert s_null == 200 and s_zero == 200

    out_null = np.asarray(d_null["rollout"], dtype=np.float32)
    out_zero = np.asarray(d_zero["rollout"], dtype=np.float32)
    _assert_valid_rollout(out_null, expected_steps=10)
    _assert_valid_rollout(out_zero, expected_steps=10)

    max_diff = np.max(np.abs(out_zero - out_null), axis=(0, 2, 3))
    assert np.all(max_diff < 1e-3 * channel_std)


@pytest.mark.parametrize("steps", [1, 30])
def test_simulate_variable_steps(server_url: str, steps: int):
    init_states, doy_sincos, _ = _make_synthetic_inputs(seed=30 + steps)
    payload = {
        "init_states": init_states.tolist(),
        "doy_sincos": doy_sincos,
        "steps": steps,
        "perturbation": None,
    }
    status, data = _post_json(f"{server_url}/simulate", payload)
    assert status == 200
    rollout = np.asarray(data["rollout"], dtype=np.float32)
    _assert_valid_rollout(rollout, expected_steps=steps)


def test_wrong_shapes_error_and_server_stays_alive(server_url: str):
    init_states, doy_sincos, _ = _make_synthetic_inputs(seed=40)

    bad_payloads = [
        {
            "init_states": init_states.tolist(),
            "doy_sincos": doy_sincos,
            "steps": 10,
            "perturbation": np.zeros((5, 32, 36), dtype=np.float32).tolist(),
        },
        {
            "init_states": [],
            "doy_sincos": doy_sincos,
            "steps": 10,
            "perturbation": None,
        },
        {
            "init_states": np.zeros((3, 5, 32, 36), dtype=np.float32).tolist(),
            "doy_sincos": doy_sincos,
            "steps": 10,
            "perturbation": np.zeros((6, 32, 36), dtype=np.float32).tolist(),
        },
    ]

    for bad in bad_payloads:
        status, _ = _post_json(f"{server_url}/simulate", bad)
        assert status >= 400

    # Verify server is still healthy and responsive
    h_status, h_data = _post_json(f"{server_url}/health", {})
    assert h_status == 200
    assert isinstance(h_data, dict) and h_data.get("status") == "ready"

    ok_payload = {
        "init_states": init_states.tolist(),
        "doy_sincos": doy_sincos,
        "steps": 1,
        "perturbation": None,
    }
    ok_status, ok_data = _post_json(f"{server_url}/simulate", ok_payload)
    assert ok_status == 200
    _assert_valid_rollout(np.asarray(ok_data["rollout"], dtype=np.float32), expected_steps=1)
