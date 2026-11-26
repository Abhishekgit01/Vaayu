"""Checkpoint save/load containing config, weights, normalization constants, and static channels."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import torch

from model.net import MonsoonNet


def save_checkpoint(path: str | Path, model: MonsoonNet, config: dict[str, Any]) -> None:
    """Save model state_dict (weights, norm buffers, static buffers) and config to a single file."""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": dict(config),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, ckpt_path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> MonsoonNet:
    """Load a MonsoonNet checkpoint and return it in eval mode."""
    ckpt_path = Path(path)
    payload = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    config = payload["config"]
    model = MonsoonNet(hidden_dim=int(config.get("hidden_dim", 64)))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
