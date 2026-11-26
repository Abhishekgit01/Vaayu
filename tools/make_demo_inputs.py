"""Create a single scenario JSON file for the dashboard from a climate NetCDF dataset."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

# Allow running directly as `python tools/make_demo_inputs.py` as well as `python -m tools.make_demo_inputs`
_REPO_ROOT_FALLBACK = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FALLBACK))

from monsoon_r1.contract import CHANNELS, NLAT, NLON, T_IN
from train.data import load_dataset


def find_repo_root(script_path: Path | None = None) -> Path:
    """Return the resolved root directory of the git repository containing this script."""
    anchor = (script_path or Path(__file__)).resolve()
    search_dir = anchor if anchor.is_dir() else anchor.parent

    try:
        proc = subprocess.run(
            ["git", "-C", str(search_dir), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return Path(proc.stdout.strip()).resolve()
    except OSError:
        pass

    for parent in (search_dir, *search_dir.parents):
        if (parent / ".git").exists():
            return parent.resolve()

    return _REPO_ROOT_FALLBACK.resolve()


def _is_within(candidate: Path, root: Path) -> bool:
    """Return True if candidate is equal to or inside root (case-normalized on Windows)."""
    cand_norm = Path(os.path.normcase(str(candidate)))
    root_norm = Path(os.path.normcase(str(root)))
    return cand_norm == root_norm or cand_norm.is_relative_to(root_norm)


def validate_out_outside_repo(out_path: Path, repo_root: Path | None = None) -> Path:
    """Refuse any output path located inside the git repo containing this script."""
    root = (repo_root or find_repo_root()).resolve()
    expanded = out_path.expanduser()
    logical_abs = Path(os.path.abspath(str(expanded)))
    resolved = expanded.resolve()

    if _is_within(logical_abs, root) or _is_within(resolved, root):
        raise ValueError(
            f"Refusing to write inside the git repository ({root}): {out_path}"
        )
    return resolved


def select_window_index(n_time: int, position: float, t_in: int = T_IN) -> int:
    """Map position in [0, 1] onto valid starting indices [0, n_time - t_in]."""
    if not math.isfinite(position) or position < 0.0 or position > 1.0:
        raise ValueError(f"--position must be a finite number in [0, 1], got {position}")
    if n_time < t_in:
        raise ValueError(f"Dataset has {n_time} time steps, need at least {t_in}")
    max_start = n_time - t_in
    start = int(round(position * max_start))
    return max(0, min(max_start, start))


def build_scenario(dataset_path: Path, position: float) -> dict:
    """Load 3 consecutive days at `position` in physical units and compute last-day doy_sincos."""
    ds = load_dataset(dataset_path)
    n_time = int(ds.data.shape[0])
    expected_tail = (len(CHANNELS), NLAT, NLON)
    if ds.data.ndim != 4 or ds.data.shape[1:] != expected_tail:
        raise ValueError(f"Unexpected dataset shape {ds.data.shape}, expected (T, {expected_tail})")

    start = select_window_index(n_time, position, t_in=T_IN)
    window = ds.data[start : start + T_IN]  # [3, 6, 32, 36], oldest first, physical units

    if not np.all(np.isfinite(window)):
        raise ValueError("Selected init_states window contains non-finite values")

    last_doy = int(ds.doy[start + T_IN - 1])
    angle = 2.0 * math.pi * float(last_doy) / 365.0
    doy_sincos = [float(math.sin(angle)), float(math.cos(angle))]
    if not all(math.isfinite(v) for v in doy_sincos):
        raise ValueError("Computed doy_sincos contains non-finite values")

    return {
        "init_states": window.tolist(),
        "doy_sincos": doy_sincos,
    }


def write_scenario(dataset_path: Path, out_path: Path, position: float) -> Path:
    """Validate paths, extract scenario, and write strict finite JSON outside the repo."""
    resolved_out = validate_out_outside_repo(out_path)
    payload = build_scenario(dataset_path, position)

    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    with resolved_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, allow_nan=False)

    return resolved_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract a 3-day scenario JSON file for the dashboard from a NetCDF dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to input NetCDF dataset (required, no default).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON file path outside the git repository (required, no default).",
    )
    parser.add_argument(
        "--position",
        type=float,
        required=True,
        help="Normalized position in [0, 1] along the dataset time axis (required).",
    )
    args = parser.parse_args(argv)

    try:
        written_path = write_scenario(args.dataset, args.out, args.position)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote scenario JSON to {written_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
