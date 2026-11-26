"""Fine-tune MonsoonNet with t2m drift fix, synthetic shocks, and propagation/locality losses."""
from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.checkpoint import load_checkpoint, save_checkpoint
from monsoon_r1.contract import NLAT, NLON, lat_weights
from monsoon_r1.scoring import STD_DEFAULT
from train.data import compute_time_split, load_dataset
from train.dataset import MonsoonWindowDataset
from train.evaluate import evaluate_model, prepare_validation_references, print_validation_table, rollout_differentiable
from train.losses import channel_weighted_forecast_loss, shock_causality_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _dilate_mask_np(mask: np.ndarray, dilate: int = 3) -> np.ndarray:
    out = mask.copy()
    for _ in range(dilate):
        o = out.copy()
        o[1:] |= out[:-1]
        o[:-1] |= out[1:]
        o[:, 1:] |= out[:, :-1]
        o[:, :-1] |= out[:, 1:]
        out = o
    return out


def sample_synthetic_shocks(
    last_states: torch.Tensor,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample random localized box/Gaussian shocks (never judge's exact boxes/values).

    Returns:
        pert: [B, 6, 32, 36] physical deltas
        near_masks: [B, 32, 36] float mask (1 within 3 cells of perturbation region)
        dist_grids: [B, 32, 36] Euclidean cell distance from perturbation region
    """
    bsz = last_states.shape[0]
    last_np = last_states.detach().cpu().numpy()
    pert = np.zeros_like(last_np, dtype=np.float32)
    near_masks = np.zeros((bsz, NLAT, NLON), dtype=np.float32)
    dist_grids = np.zeros((bsz, NLAT, NLON), dtype=np.float32)

    ii, jj = np.indices((NLAT, NLON))

    for b in range(bsz):
        ci = int(rng.integers(3, NLAT - 3))
        cj = int(rng.integers(3, NLON - 3))
        half_h = int(rng.integers(1, 7))   # size 3..13
        half_w = int(rng.integers(1, 7))
        use_gaussian = bool(rng.integers(0, 2))

        if use_gaussian:
            sigma = float(rng.uniform(1.5, 4.0))
            d2 = (ii - ci) ** 2 + (jj - cj) ** 2
            profile = np.exp(-0.5 * d2 / (sigma ** 2)).astype(np.float32)
            box = d2 <= (max(2, int(round(sigma * 1.5))) ** 2)
            profile = np.where(box, profile, 0.0).astype(np.float32)
        else:
            i0, i1 = max(0, ci - half_h), min(NLAT, ci + half_h + 1)
            j0, j1 = max(0, cj - half_w), min(NLON, cj + half_w + 1)
            box = np.zeros((NLAT, NLON), dtype=bool)
            box[i0:i1, j0:j1] = True
            profile = box.astype(np.float32)

        # Randomize which channels are shocked: thermal/moisture vs wind (or both)
        mode = int(rng.integers(0, 3))
        if mode in (0, 2):
            dt2m = float(rng.uniform(-4.0, 4.0))
            ftcwv = float(rng.uniform(-0.20, 0.20))
            pert[b, 0] = dt2m * profile
            pert[b, 5] = ftcwv * last_np[b, 5] * profile
        if mode in (1, 2):
            fu = float(rng.uniform(-0.50, 0.50))
            fv = float(rng.uniform(-0.50, 0.50))
            pert[b, 2] = fu * last_np[b, 2] * profile
            pert[b, 3] = fv * last_np[b, 3] * profile

        near_masks[b] = _dilate_mask_np(box, dilate=3).astype(np.float32)
        bi, bj = np.where(box)
        dist_grids[b] = np.sqrt(((ii[..., None] - bi) ** 2 + (jj[..., None] - bj) ** 2)).min(axis=-1).astype(np.float32)

    dev = last_states.device
    return (
        torch.from_numpy(pert).to(dev),
        torch.from_numpy(near_masks).to(dev),
        torch.from_numpy(dist_grids).to(dev),
    )


def train_shock_run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))

    ds = load_dataset(args.dataset)
    splits = compute_time_split(ds.data.shape[0])
    train_data = ds.data[splits.train]
    train_doy = ds.doy[splits.train]
    val_data = ds.data[splits.val]
    val_doy = ds.doy[splits.val]

    std_np = np.std(train_data, axis=(0, 2, 3)).astype(np.float32)
    std_np = np.where(std_np > 1e-6, std_np, STD_DEFAULT).astype(np.float32)
    std_phys = torch.from_numpy(std_np).to(device)

    model = load_checkpoint(args.checkpoint, map_location=device).to(device)
    w_lat = torch.tensor(lat_weights(), dtype=torch.float32, device=device)
    channel_weights = torch.tensor([args.w_t2m, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    horizon = args.horizon
    train_ds = MonsoonWindowDataset(train_data, train_doy, horizon=horizon)
    val_refs = prepare_validation_references(
        train_data, train_doy, val_data, val_doy, max_windows=16 if args.smoke else None
    )

    best_score = -float("inf")
    best_eval = None
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        gen = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=gen, drop_last=False)

        model.train()
        sums = {"mse": 0.0, "rain_extra": 0.0, "grad_l1": 0.0, "var_match": 0.0, "tcwv_cons": 0.0, "shock_total": 0.0, "total": 0.0}
        n_batches = 0

        for init_b, doy_b, target_b in loader:
            init_b = init_b.to(device)
            doy_b = doy_b.to(device)
            target_b = target_b.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_norm, _ = rollout_differentiable(model, init_b, doy_b, steps=horizon, noise_std=args.noise_std)
            target_norm = model.norm.normalize(target_b)
            loss_fc, fc_terms = channel_weighted_forecast_loss(
                pred_norm,
                target_norm,
                target_b[:, :, 1],
                w_lat=w_lat,
                channel_weights=channel_weights,
                w_mse=args.w_mse,
                w_rain_extra=args.w_rain_extra,
                w_grad=args.w_grad,
                w_var=args.w_var,
                w_tcwv_cons=args.w_tcwv_cons,
            )

            shock_val = 0.0
            total_loss = loss_fc
            if float(rng.random()) < args.shock_prob:
                # Sub-slice to half batch for paired (base, shocked) rollout to cap peak RAM
                sb = max(1, init_b.shape[0] // 2)
                init_sub = init_b[:sb]
                doy_sub = doy_b[:sb]
                pert, near_m, dist_g = sample_synthetic_shocks(init_sub[:, -1], rng)

                init_shocked = init_sub.clone()
                init_shocked[:, -1] = init_shocked[:, -1] + pert
                init_shocked[:, -1, 1] = torch.clamp(init_shocked[:, -1, 1], min=0.0)

                _, base_phys = rollout_differentiable(model, init_sub, doy_sub, steps=horizon, noise_std=0.0)
                _, shock_phys = rollout_differentiable(model, init_shocked, doy_sub, steps=horizon, noise_std=0.0)
                loss_sh, sh_terms = shock_causality_loss(
                    base_phys,
                    shock_phys,
                    near_masks=near_m,
                    dist_grids=dist_g,
                    std_phys=std_phys,
                    w_local=args.w_local,
                    w_prop=args.w_prop,
                    w_stable=args.w_stable,
                    w_tele=args.w_tele,
                )
                total_loss = total_loss + args.w_shock * loss_sh
                shock_val = sh_terms["shock_total"]

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in ("mse", "rain_extra", "grad_l1", "var_match", "tcwv_cons"):
                sums[k] += fc_terms[k]
            sums["shock_total"] += shock_val
            sums["total"] += float(total_loss.detach().item())
            n_batches += 1
            if args.smoke and n_batches >= 2:
                break

        scheduler.step()
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | time={dt:.2f}s | lr={scheduler.get_last_lr()[0]:.2e} | "
            f"mse={avg['mse']:.4f} grad={avg['grad_l1']:.4f} var={avg['var_match']:.4f} "
            f"tcwv_c={avg['tcwv_cons']:.4f} shock={avg['shock_total']:.4f} total={avg['total']:.4f}"
        )

        eval_res = evaluate_model(model, val_refs, device=device, batch_size=max(args.batch_size, 32))
        print_validation_table(eval_res, epoch=epoch)

        if best_eval is None or eval_res["selection_score"] > best_score:
            best_score = eval_res["selection_score"]
            best_eval = eval_res
            save_checkpoint(args.out, model, {"hidden_dim": model.hidden_dim})
            print(f"Saved best shock-tuned checkpoint -> {args.out} (score={best_score:+.4f})")

        gc.collect()

    return {"best_score": best_score, "best_eval": best_eval, "avg_epoch_s": float(np.mean(epoch_times))}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune MonsoonNet for t2m skill and shock propagation.")
    p.add_argument("--dataset", required=True, help="Path to india_climate_train.nc (required).")
    p.add_argument("--checkpoint", required=True, help="Initial checkpoint path (required).")
    p.add_argument("--out", required=True, help="Output checkpoint path (required).")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--noise-std", type=float, default=0.02)
    p.add_argument("--shock-prob", type=float, default=0.5)
    p.add_argument("--w-t2m", type=float, default=2.5)
    p.add_argument("--w-mse", type=float, default=1.0)
    p.add_argument("--w-rain-extra", type=float, default=4.0)
    p.add_argument("--w-grad", type=float, default=0.1)
    p.add_argument("--w-var", type=float, default=0.1)
    p.add_argument("--w-tcwv-cons", type=float, default=0.05)
    p.add_argument("--w-shock", type=float, default=1.0)
    p.add_argument("--w-local", type=float, default=1.0)
    p.add_argument("--w-prop", type=float, default=2.0)
    p.add_argument("--w-stable", type=float, default=0.5)
    p.add_argument("--w-tele", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    train_shock_run(parse_args())
