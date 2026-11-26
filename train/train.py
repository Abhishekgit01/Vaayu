"""Training script for MonsoonNet with curriculum rollout, noise injection, and checkpointing."""
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
from model.net import MonsoonNet
from train.data import compute_time_split, load_dataset
from train.dataset import MonsoonWindowDataset, compute_train_stats
from train.evaluate import evaluate_model, prepare_validation_references, print_validation_table, rollout_differentiable
from train.losses import MonsoonLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def curriculum_horizon(epoch_idx: int, total_epochs: int, start_horizon: int = 1) -> int:
    """Switch rollout horizon across epochs starting from start_horizon."""
    frac = epoch_idx / max(total_epochs, 1)
    if start_horizon >= 4:
        return 4 if frac < 0.50 else 8
    if frac < 0.25:
        return 1
    if frac < 0.50:
        return 2
    if frac < 0.75:
        return 4
    return 8


def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def train_run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)

    ds = load_dataset(args.dataset)
    splits = compute_time_split(ds.data.shape[0])
    train_data = ds.data[splits.train]
    train_doy = ds.doy[splits.train]
    val_data = ds.data[splits.val]
    val_doy = ds.doy[splits.val]

    mean, std = compute_train_stats(train_data)
    if args.resume:
        model = load_checkpoint(args.resume, map_location=device).to(device)
    else:
        model = MonsoonNet(
            static=torch.from_numpy(ds.static),
            mean=torch.from_numpy(mean),
            std=torch.from_numpy(std),
            hidden_dim=args.width,
        ).to(device)

    ch_weights = None
    if args.channel_weights:
        ch_weights = [float(x.strip()) for x in args.channel_weights.split(",")]
        if len(ch_weights) != 6:
            raise ValueError(f"--channel-weights must have 6 values, got {len(ch_weights)}")

    criterion = MonsoonLoss(
        w_mse=args.w_mse,
        w_rain_extra=args.w_rain_extra,
        w_grad=args.w_grad,
        w_var=args.w_var,
        channel_weights=ch_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    train_ds = MonsoonWindowDataset(train_data, train_doy, horizon=1)
    val_refs = prepare_validation_references(
        train_data, train_doy, val_data, val_doy, max_windows=16 if args.smoke else None
    )

    best_score = -float("inf")
    best_eval = None
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        horizon = curriculum_horizon(epoch - 1, args.epochs, start_horizon=args.start_horizon)
        train_ds.set_horizon(horizon)

        gen = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=gen,
            drop_last=False,
        )

        model.train()
        sums = {"mse": 0.0, "rain_extra": 0.0, "grad_l1": 0.0, "var_match": 0.0, "total": 0.0}
        n_batches = 0

        for init_b, doy_b, target_b in loader:
            init_b = init_b.to(device)
            doy_b = doy_b.to(device)
            target_b = target_b.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_norm, _ = rollout_differentiable(
                model, init_b, doy_b, steps=horizon, noise_std=args.noise_std
            )
            target_norm = model.norm.normalize(target_b)
            loss, terms = criterion(pred_norm, target_norm, target_b[:, :, 1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in sums:
                sums[k] += terms[k]
            n_batches += 1
            if args.smoke and n_batches >= 2:
                break

        scheduler.step()
        dt = time.perf_counter() - t0
        epoch_times.append(dt)

        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | horizon={horizon} | time={dt:.2f}s | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"mse={avg['mse']:.4f} rain_extra={avg['rain_extra']:.4f} "
            f"grad_l1={avg['grad_l1']:.4f} var_match={avg['var_match']:.4f} total={avg['total']:.4f}"
        )

        eval_res = evaluate_model(model, val_refs, device=device, batch_size=max(args.batch_size, 64))
        print_validation_table(eval_res, epoch=epoch)

        if eval_res["selection_score"] > best_score:
            best_score = eval_res["selection_score"]
            best_eval = eval_res
            save_checkpoint(args.out, model, {"hidden_dim": args.width})
            print(f"Saved best checkpoint -> {args.out} (score={best_score:+.4f})")

        gc.collect()

    return {
        "best_score": best_score,
        "best_eval": best_eval,
        "avg_epoch_s": float(np.mean(epoch_times)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MonsoonNet with multi-step rollout curriculum.")
    p.add_argument("--dataset", required=True, help="Path to india_climate_train.nc (required, no default).")
    p.add_argument("--out", required=True, help="Output checkpoint path (required, no default).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--noise-std", type=float, default=0.02)
    p.add_argument("--w-mse", type=float, default=1.0)
    p.add_argument("--w-rain-extra", type=float, default=4.0)
    p.add_argument("--w-grad", type=float, default=0.2)
    p.add_argument("--w-var", type=float, default=0.1)
    p.add_argument("--channel-weights", default=None, help="Comma-separated 6 per-channel MSE weights.")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    p.add_argument("--start-horizon", type=int, default=1, help="Minimum rollout horizon.")
    p.add_argument("--smoke", action="store_true", help="Tiny run of a few batches for tests.")
    return p.parse_args(argv)


if __name__ == "__main__":
    train_run(parse_args())
