# Monsoon Twin — Yugaantar DevForge (Track 2: Multiverse of Monsoon)

**Monsoon Twin** is a fast, offline-only, physics-aware digital twin of India's daily weather (`32 × 36` grid, `1°` resolution, `lat 6.5..37.5`, `lon 65.5..100.5`) serving 10+ day autoregressive rollouts and localized shock simulations over `POST /simulate`.

## Architecture (`MonsoonNet`, 125,126 Parameters)

`model/net.py` implements a custom **125,126-parameter** fully convolutional residual network (`MonsoonNet`) designed for causal, local-first perturbation propagation on CPU:

- **Inputs (22 planes per step):** 3 consecutive days × 6 dynamic channels (`t2m`, `tp`, `u10`, `v10`, `msl`, `tcwv`) + 2 registered static buffers (`land_sea_mask`, `orography_m`) + 2 broadcast day-of-year phase planes (`sin`, `cos`).
- **Exact 3-cell receptive field:** A `1×1` stem, **exactly three** `3×3` `reflect`-padded residual blocks (`_LocalResBlock`), and a `1×1` head—with no pooling, no strided or dilated convolutions, no attention, and no global operations. By construction, any single-cell perturbation affects at most a 3-cell radius on Day 1, ensuring physical locality before spreading outward on later days.
- **Physical-unit I/O & Normalization:** `model/normalize.py` applies per-channel z-scoring (`tp` transformed via `log1p` / `expm1` and clamped `>= 0`) inside the model using buffers saved in the checkpoint. `model/rollout.py` runs deterministically (`eval()`, `torch.no_grad()`), adds any physical perturbation to `init_states[-1]` before normalization, rotates `(sin, cos)` by `2π/365` per step, and clamps `t2m` to `[233.15, 333.15]` K.

## Setup (Offline Serving)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the offline server on `0.0.0.0:8000`:

```bash
python serve.py --port 8000
```

Validate against the Round 1 API contract:

```bash
python -m monsoon_r1.schema_validator http://localhost:8000
```

## Training & Evaluation

Install training dependencies (in addition to `requirements.txt`):

```bash
pip install -r requirements-train.txt
```

1. **Baseline evaluation** (persistence & on-the-fly training-split DOY climatology):
   ```bash
   python train/eval_skill.py --dataset <PATH_TO_NETCDF>
   ```
2. **Curriculum rollout training** (`1 -> 2 -> 4 -> 8` steps, noise injection `0.02`, latitude-weighted MSE + heavy-rain weighting + gradient L1 + variance matching):
   ```bash
   python -m train.train --dataset <PATH_TO_NETCDF> --out weights/monsoon.pt --epochs 16 --width 64 --lr 1e-3 --batch-size 32 --device auto
   ```
3. **Shock & `t2m` fine-tuning** (synthetic localized shocks, Day-1 locality & Days 4–8 propagation losses, `tcwv` conservation):
   ```bash
   python -m train.train_shock --dataset <PATH_TO_NETCDF> --checkpoint weights/monsoon.pt --out weights/monsoon_shock.pt --epochs 4 --horizon 8 --lr 2e-4 --batch-size 16 --w-t2m 2.5 --w-grad 0.1 --w-tcwv-cons 0.05 --shock-prob 0.5 --device auto
   ```
4. **Judge-style shock evaluation**:
   ```bash
   python train/shock_eval.py --checkpoint weights/monsoon.pt --dataset <PATH_TO_NETCDF> --n-windows 10 --steps 8
   ```
5. **Unit tests & repo hygiene scan**:
   ```bash
   python -m pytest -v tests/
   python tools/scan_repo.py --history .
   ```

## Known Limitations

- **`t2m` Day-4 RMSE:** While `tp` heavy-rain CSI beats the baselines at Days 1, 4, and 8 and all Day-8 spatial variance ratios sit within `0.40..2.50`, `t2m` Day-4 RMSE (`~3.57 K` after fine-tuning) misses persistence (`1.95 K`) and climatology (`1.79 K`) due to diurnal/thermal-cycle drift across multi-step rollouts.
- **Shock `propagates` threshold:** Under `monsoon_r1.scoring.causality_checks`, Day-1 locality (`100%`), stability (`< 5 std`), null-shock exactness (`0.0`), and front speed (`<= 8 cells/day`) pass, but `propagates` remains `False` because outside-halo energy on Days 4–8 does not yet exceed the `2%` peak-energy threshold across all windows.

## Repository Layout

- `serve.py` — Offline FastAPI inference server (`POST /health`, `POST /simulate`)
- `model/` — `MonsoonNet`, `ChannelNormalizer`, `rollout`, and checkpoint utilities
- `train/` — Data splitting, baselines, curriculum training, shock fine-tuning, and evaluation
- `weights/` — Trained checkpoints tracked via Git LFS (`weights/*.pt`, `weights/*.onnx`)
- `tests/` — Seeded synthetic unit tests (zero saved data)
- `tools/` — Local utility & repository hygiene scanner (`scan_repo.py`)
- `dashboard/` — Frontend application (separate from server)
- `docs/` — API contract and integration log (`docs/INTEGRATION.md`)
