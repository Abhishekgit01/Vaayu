# Monsoon Twin — Yugaantar DevForge (Track 2: Multiverse of Monsoon)

A fast, physics-aware digital twin of India's daily weather serving 10+ day rollouts and localized shock simulations over `POST /simulate`.

## Repository Layout

- `serve.py` — Inference server (`POST /health`, `POST /simulate`), offline-only
- `model/` — Model architecture, normalization, rollout, and shock handling
- `train/` — Data loading, training loop, and evaluation scripts
- `weights/` — Trained model weights (Git LFS: `*.pt`, `*.onnx`)
- `tests/` — Contract, zero-shock, locality, stability, and speed tests
- `tools/` — Local utility scripts (never writes data into the repo)
- `dashboard/` — Frontend dashboard (separate from server)
- `docs/` — Specifications and integration notes (`docs/INTEGRATION.md`)

## Setup & Verification

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the server:

```bash
python serve.py --port 8000
```

Validate against the Round 1 API contract:

```bash
python -m monsoon_r1.schema_validator http://localhost:8000
```

## Training

_To be documented once the training pipeline is implemented._
