"""Starter server for Round 1. This is API plumbing only: `simulate()` below is a placeholder (it repeats the last day).
Replace it with YOUR model. The judge runs this exact file in an offline container:  python serve.py --port 8000

Rules: listen on 0.0.0.0 (not 127.0.0.1), load weights from files next to this one (there is no internet), keep
the model loaded in memory (a 10-step rollout must return in under 5 s), return finite numbers in physical units.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from model.checkpoint import load_checkpoint
from model.net import N_CHANNELS, NLAT, NLON, T_IN
from model.rollout import rollout as model_rollout

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_NAME = "monsoon-twin"

MAX_PAYLOAD_BYTES = 16 * 1024 * 1024  # 16 MB request cap
MAX_STEPS = 365

# Per-channel physical bounds on init_states: (min, max) derived from training range + wide margin
# Channels: 0=t2m (K), 1=tp (mm/d), 2=u10 (m/s), 3=v10 (m/s), 4=msl (Pa), 5=tcwv (kg/m2)
STATE_BOUNDS = (
    (180.0, 360.0),
    (-1e-3, 2000.0),
    (-150.0, 150.0),
    (-150.0, 150.0),
    (70000.0, 125000.0),
    (-1e-3, 250.0),
)
# Per-channel max |perturbation| delta in physical units
PERT_MAX_ABS = (100.0, 1000.0, 100.0, 100.0, 20000.0, 150.0)
# Physical per-channel standard deviations from stats.json for z-score ensemble noise
CHANNEL_STD_PHYS = (
    13.46363353729248,
    7.553342342376709,
    2.8989806175231934,
    2.4269521236419678,
    619.0198364257812,
    20.49437141418457,
)

torch.set_num_threads(4)
_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "monsoon.pt"
_MODEL = load_checkpoint(_WEIGHTS_PATH, map_location="cpu")
_MODEL.eval()

# Warm-up rollout at start-up so /health reports ready only after cold-start completes
_WARMUP_INIT = np.zeros((T_IN, N_CHANNELS, NLAT, NLON), dtype=np.float32)
_WARMUP_INIT[:, 0] = 295.0
_WARMUP_INIT[:, 4] = 101325.0
_WARMUP_INIT[:, 5] = 30.0
with torch.no_grad():
    _ = model_rollout(_MODEL, _WARMUP_INIT, [0.0, 1.0], steps=1, perturbation=None)


def _arr(x):
    """Accept nested lists (the contract) or {'b64': ..., 'shape': [...]} (tolerated)."""
    if isinstance(x, dict):
        import base64
        raw = base64.b64decode(x["b64"], validate=True)
        shape = tuple(int(s) for s in x["shape"])
        return np.frombuffer(raw, dtype="<f4").reshape(shape).astype("f4")
    return np.asarray(x, dtype="f4")


def _validate_inputs(init: np.ndarray, doy_sc: np.ndarray, steps_raw, pert: np.ndarray | None) -> int:
    if init.shape != (T_IN, N_CHANNELS, NLAT, NLON) or not np.isfinite(init).all():
        raise HTTPException(status_code=422, detail="Invalid init_states shape or non-finite values.")
    for ci, (lo, hi) in enumerate(STATE_BOUNDS):
        cmin = float(init[:, ci].min())
        cmax = float(init[:, ci].max())
        if cmin < lo or cmax > hi:
            raise HTTPException(
                status_code=422,
                detail=f"init_states channel {ci} out of physical bounds [{lo}, {hi}].",
            )
    if doy_sc.shape != (2,) or not np.isfinite(doy_sc).all() or float(np.abs(doy_sc).max()) > 1.5:
        raise HTTPException(status_code=422, detail="Invalid doy_sincos shape or values.")
    if isinstance(steps_raw, bool) or not isinstance(steps_raw, int) or steps_raw < 1 or steps_raw > MAX_STEPS:
        raise HTTPException(status_code=422, detail=f"steps must be an integer in [1, {MAX_STEPS}].")
    if pert is not None:
        if pert.shape != (N_CHANNELS, NLAT, NLON) or not np.isfinite(pert).all():
            raise HTTPException(status_code=422, detail="Invalid perturbation shape or non-finite values.")
        for ci, max_abs in enumerate(PERT_MAX_ABS):
            if float(np.abs(pert[ci]).max()) > max_abs:
                raise HTTPException(
                    status_code=422,
                    detail=f"perturbation channel {ci} exceeds max magnitude {max_abs}.",
                )
    return int(steps_raw)


def simulate_ensemble(
    init_states: np.ndarray, doy_sincos, steps: int, perturbation
) -> tuple[np.ndarray, np.ndarray]:
    """Run a 5-member ensemble (1 control + 4 microscopic Gaussian-perturbed members).

    Returns (mean [steps,6,32,36], ensemble_spread std [steps,6,32,36]) in physical units.
    """
    base_init = np.asarray(init_states, dtype=np.float32).copy()
    if perturbation is not None:
        pert = np.asarray(perturbation, dtype=np.float32)
        base_init[-1] = base_init[-1] + pert
        base_init[-1, 1] = np.maximum(base_init[-1, 1], 0.0)

    base_t = torch.from_numpy(base_init)
    noise_scale = torch.tensor(CHANNEL_STD_PHYS, dtype=base_t.dtype).view(1, N_CHANNELS, 1, 1) * 0.015
    gen = torch.Generator().manual_seed(42)
    members = []

    with torch.no_grad():
        for member_idx in range(5):
            if member_idx == 0:
                member_init = base_init
            else:
                noise = torch.randn(base_t.shape, generator=gen, dtype=base_t.dtype) * noise_scale
                member_t = base_t + noise
                member_t[:, 1].clamp_(min=0.0)
                member_init = member_t.numpy()
            members.append(
                model_rollout(_MODEL, member_init, doy_sincos, steps=steps, perturbation=None)
            )

    stacked = np.stack(members, axis=0).astype(np.float32)
    return stacked.mean(axis=0), stacked.std(axis=0)


def simulate(init_states: np.ndarray, doy_sincos, steps: int, perturbation) -> np.ndarray:
    """init_states [3,6,32,36] physical units, oldest first. perturbation [6,32,36] delta or None.
    Return [steps,6,32,36] physical units. doy_sincos = [sin, cos] of the day of year of init_states[-1]."""
    mean, _ = simulate_ensemble(init_states, doy_sincos, steps=steps, perturbation=perturbation)
    return mean


@app.post("/health")
def health():
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    import time
    t0 = time.perf_counter()

    content_len = request.headers.get("content-length")
    if content_len is not None:
        try:
            if int(content_len) > MAX_PAYLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Request payload too large.")
        except ValueError:
            pass

    try:
        raw_body = await request.body()
        if len(raw_body) > MAX_PAYLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Request payload too large.")
        if not raw_body.strip():
            raise HTTPException(status_code=400, detail="Empty request body.")
        body = json.loads(raw_body)
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object.")
        init = _arr(body["init_states"])
        doy_sc = np.asarray(body["doy_sincos"], dtype="f4")
        pert = None if body.get("perturbation") is None else _arr(body["perturbation"])
        steps = _validate_inputs(init, doy_sc, body["steps"], pert)
    except HTTPException:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Malformed simulation request.")

    try:
        mean, spread = simulate_ensemble(init, doy_sc, steps, pert)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Simulation failed physical bounds: {exc}")

    if not (np.isfinite(mean).all() and np.isfinite(spread).all()):
        raise HTTPException(status_code=422, detail="Simulation produced non-finite values.")

    return {
        "rollout": np.asarray(mean, dtype="f4").tolist(),
        "ensemble_spread": np.asarray(spread, dtype="f4").tolist(),
        "inference_ms": (time.perf_counter() - t0) * 1000.0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, log_level="warning")
