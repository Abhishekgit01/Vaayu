# Integration Contract & Status

## Endpoints and shapes

Grid: 32 x 36 cells of 1 deg, centres lat 6.5..37.5 (south to north), lon 65.5..100.5 (west to east). Each cell is the mean of the 4 surrounding 1 deg ERA5 points. Channel order, axis 1 of every state array:

| i | name | meaning | unit |
|---|---|---|---|
| 0 | t2m | 2 m temperature | K |
| 1 | tp | daily total precipitation | mm/day |
| 2 | u10 | 10 m eastward wind | m/s |
| 3 | v10 | 10 m northward wind | m/s |
| 4 | msl | mean sea level pressure | Pa |
| 5 | tcwv | total column water vapour | kg/m2 |

Daily fields: `t2m`, `u10`, `v10`, `msl`, `tcwv` are the mean of the 00, 06, 12, 18 UTC samples; `tp` is the sum over the 24 h of the day.
Normalisation statistics (mean, std per channel) ship in `stats.json` for your own use; the wire is always physical.

- `POST /health` -> `200 {"status": "ready", "model_name": "..."}` (anything else = not ready).
- `POST /simulate`, JSON body:
  ```json
  {"init_states": [3,6,32,36], "doy_sincos": [sin, cos], "steps": 10, "perturbation": null | [6,32,36]}
  ```
  `init_states` are 3 consecutive days, oldest first; the forecast starts the day after the last one. `doy_sincos = [sin(2 pi doy/365), cos(2 pi doy/365)]` for the LAST input day. `perturbation` is a delta in physical units added to `init_states[-1]` before you simulate (`null` = none).
  Response: `{"rollout": [steps,6,32,36], "inference_ms": float}`; `rollout[k]` is day `k+1` after the last input day.
- **Wire format (one canonical format):** Plain nested JSON lists of finite floats, in both directions (`init_states` `3 x 6 x 32 x 36`, `perturbation` `6 x 32 x 36` or `null`, `rollout` `steps x 6 x 32 x 36`). NaN and Inf are not valid JSON and fail the validity gate.
- **Units:** Physical, always (`K`, `mm/day`, `m/s`, `m/s`, `Pa`, `kg/m2`).
- **Limits:** 10 steps in under 5.0 s wall time, response under 80 MB, no redirects.

## What exists

- **Backend:** Repo scaffolded with the unchanged starter `serve.py` (`model_name: "placeholder-persistence"`), `requirements.txt`, `.gitignore`, `.gitattributes`, and empty `model/`, `train/`, `tests/`, `tools/`, `weights/` directories.
- **Frontend:** Empty `dashboard/` directory (`.gitkeep` only). No frontend code yet.

## Requests to the other side

- **Frontend -> Backend:** Confirm if CORS middleware is needed on `serve.py` for local/browser preview (separate scoped server task).
- **Backend -> Frontend:** Ensure `src/api/client.ts` targets configurable base URL defaulting to `http://localhost:8000` and handles `POST /health` and `POST /simulate` with physical units.

## Known issues

- `serve.py` currently uses the starter persistence placeholder (`repeat(last, steps)`), not a trained model.
- `serve.py` does not yet include CORS headers for cross-origin browser previews.

## Log

- **2026-11-26:** Initialized repo scaffold (`serve.py` copied from `monsoon_r1/template/serve.py`, `.gitignore`, `.gitattributes`, `requirements.txt`, `README.md`, `docs/INTEGRATION.md`, and directory `.gitkeep` files). Verified with `monsoon_r1.schema_validator`.
- **2026-11-26 (Task 2):** Added `model/` (`net.py`, `normalize.py`, `rollout.py`, `checkpoint.py`), `tests/test_model.py`, and `requirements-train.txt` (Python 3.11 venv with `torch==2.15.0+cpu`); moved `monsoon_data/` and venv outside the repo and initialized standalone git repo. `serve.py` still uses the starter placeholder until Task 5.
- **2026-11-26 (Task 5):** Wired `weights/monsoon.pt` (`model_name: "monsoon-twin"`) into `serve.py` with `torch.set_num_threads(4)`, start-up warm-up rollout, input shape/finiteness validation (HTTP 400/422), and `model.rollout.rollout`.
- **2026-11-26 (Forecast Fine-Tune):** Added optional `--channel-weights` to `train/losses.py` & `train/train.py`; fine-tuned `weights/monsoon.pt` for 6 epochs (`--batch-size 16 --lr 2e-4 --start-horizon 4 --w-grad 0.12 --channel-weights 2.5,1,1,1,1,1`), improving `mean_skill` to `-0.1922`, beating Heavy-Rain CSI at Days 1/4/8, and keeping all Day-8 variance ratios in `[0.80, 1.74]`.
- **2026-11-26 (Task 3):** Added training pipeline (`train/dataset.py`, `train/losses.py`, `train/evaluate.py`, `train/train.py`, `tests/test_train.py`) and trained `weights/monsoon.pt` (best checkpoint saved at epoch 15/16, `score=-0.3644`, all 6 day-8 variance ratios within `0.4..2.5`, `tp` CSI beats baseline at days 1 & 4; RMSE at days 4 & 8 still below climatology). `serve.py` still uses the starter placeholder until Task 5.
- **2026-11-26 (Task 4b):** Added `train/train_shock.py`, `tests/test_train_shock.py`, and shock/channel-weighted loss helpers in `train/losses.py`; fine-tuned 4 epochs from `weights/monsoon.pt` into `weights/monsoon_shock.pt` (`mean_skill` `-0.3644 -> -0.2975`, all 6 day-8 variance ratios `0.60..1.35`, `tp` CSI beats baseline at days 1/4/8, `warm_ocean` spread `0.616 -> 1.691 cells/d`, `propagates` still `False` so `weights/monsoon.pt` was not overwritten). `serve.py` still uses the starter placeholder until Task 5.
