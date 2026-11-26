"""Generate an external scenario JSON file for dashboard/API testing.

Usage:
    python tools/make_demo_inputs.py --dataset /path/to/data.nc --out /tmp/scenario.json --position 0.5
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monsoon_r1.contract import CHANNELS, NLAT, NLON, T_IN
from train.data import load_dataset


def find_git_root(start: Path = REPO_ROOT) -> Path:
    """Return the resolved git repository root, falling back to REPO_ROOT."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(proc.stdout.strip()).resolve()
    except (subprocess.SubprocessError, OSError):
        return REPO_ROOT.resolve()


def ensure_outside_repo(out_path: Path, repo_root: Path | None = None) -> Path:
    """Refuse to write if out_path resolves inside the git repository."""
    root = (repo_root or find_git_root()).resolve()
    resolved = out_path.resolve()
    if resolved == root or resolved.is_relative_to(root):
        raise ValueError(
            f"Refusing to write inside the git repository: {resolved} (repo root: {root})"
        )
    return resolved


def build_scenario(dataset_path: str | Path, position: float) -> dict[str, object]:
    """Extract a 3-day physical-unit scenario and last-day doy_sincos at position in [0, 1]."""
    if not math.isfinite(position) or not (0.0 <= position <= 1.0):
        raise ValueError(f"--position must be a finite float in [0, 1], got {position}")

    ds = load_dataset(dataset_path)
    n_time = ds.data.shape[0]
    if n_time < T_IN:
        raise ValueError(f"Dataset time length {n_time} is shorter than T_IN={T_IN}")

    max_start = n_time - T_IN
    start = max(0, min(max_start, int(round(position * max_start))))
    init_states = np.asarray(ds.data[start : start + T_IN], dtype=np.float32)

    expected_shape = (T_IN, len(CHANNELS), NLAT, NLON)
    if init_states.shape != expected_shape:
        raise ValueError(f"Unexpected init_states shape {init_states.shape}, expected {expected_shape}")

    last_doy = float(ds.doy[start + T_IN - 1])
    angle = 2.0 * math.pi * last_doy / 365.0
    doy_sincos = [math.sin(angle), math.cos(angle)]

    if not np.all(np.isfinite(init_states)):
        raise ValueError("Non-finite values found in init_states")
    if not all(math.isfinite(v) for v in doy_sincos):
        raise ValueError("Non-finite values found in doy_sincos")

    return {
        "init_states": init_states.tolist(),
        "doy_sincos": doy_sincos,
    }


def write_scenario(dataset_path: str | Path, out_path: str | Path, position: float) -> Path:
    """Validate output path is outside the git repo, build scenario, and write JSON."""
    resolved_out = ensure_outside_repo(Path(out_path))
    payload = build_scenario(dataset_path, position)

    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    with resolved_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    return resolved_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a scenario JSON file from a climate NetCDF dataset."
    )
    parser.add_argument("--dataset", required=True, help="Path to input NetCDF dataset.")
    parser.add_argument("--out", required=True, help="Output JSON path (must be outside git repo).")
    parser.add_argument(
        "--position",
        type=float,
        required=True,
        help="Normalized position along the dataset time axis, in [0, 1].",
    )
    args = parser.parse_args(argv)

    try:
        out_file = write_scenario(args.dataset, args.out, args.position)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote scenario JSON to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
