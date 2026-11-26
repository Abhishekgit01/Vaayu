"""Starter server for Round 1. This is API plumbing only: `simulate()` below is a placeholder (it repeats the last day).
Replace it with YOUR model. The judge runs this exact file in an offline container:  python serve.py --port 8000

Rules: listen on 0.0.0.0 (not 127.0.0.1), load weights from files next to this one (there is no internet), keep
the model loaded in memory (a 10-step rollout must return in under 5 s), return finite numbers in physical units.
"""
from pathlib import Path
import argparse
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
        return np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    return np.asarray(x, dtype="f4")


def _validate_inputs(init: np.ndarray, doy_sc: np.ndarray, steps_raw, pert: np.ndarray | None) -> int:
    if init.shape != (T_IN, N_CHANNELS, NLAT, NLON) or not np.isfinite(init).all():
        raise HTTPException(status_code=422, detail="Invalid init_states shape or non-finite values.")
    if doy_sc.shape != (2,) or not np.isfinite(doy_sc).all():
        raise HTTPException(status_code=422, detail="Invalid doy_sincos shape or non-finite values.")
    if isinstance(steps_raw, bool) or not isinstance(steps_raw, int) or steps_raw < 1:
        raise HTTPException(status_code=422, detail="steps must be a positive integer.")
    if pert is not None and (pert.shape != (N_CHANNELS, NLAT, NLON) or not np.isfinite(pert).all()):
        raise HTTPException(status_code=422, detail="Invalid perturbation shape or non-finite values.")
    return int(steps_raw)


def simulate(init_states: np.ndarray, doy_sincos, steps: int, perturbation) -> np.ndarray:
    """init_states [3,6,32,36] physical units, oldest first. perturbation [6,32,36] delta or None.
    Return [steps,6,32,36] physical units. doy_sincos = [sin, cos] of the day of year of init_states[-1]."""
    with torch.no_grad():
        return model_rollout(_MODEL, init_states, doy_sincos, steps=steps, perturbation=perturbation)


@app.post("/health")
def health():
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    import time
    t0 = time.perf_counter(); body = await request.json()
    try:
        init = _arr(body["init_states"])
        doy_sc = np.asarray(body["doy_sincos"], dtype="f4")
        pert = None if body.get("perturbation") is None else _arr(body["perturbation"])
        steps = _validate_inputs(init, doy_sc, body["steps"], pert)
    except HTTPException:
        raise
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Malformed simulation request.")
    out = simulate(init, doy_sc, steps, pert)
    return {"rollout": np.asarray(out, dtype="f4").tolist(), "inference_ms": (time.perf_counter() - t0) * 1000.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, log_level="warning")
