"""Dataset loading, time-based splitting, and sliding-window construction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import xarray as xr

from monsoon_r1.contract import CHANNELS, NLAT, NLON, T_IN

STATIC_CHANNELS = ("land_sea_mask", "orography_m")


@dataclass(frozen=True)
class ClimateDataset:
    """In-memory climate arrays loaded from a NetCDF file."""

    data: np.ndarray        # [T, 6, 32, 36], float32, physical units
    static: np.ndarray      # [2, 32, 36], float32
    doy: np.ndarray         # [T], int32 day-of-year (1..366)
    lat: np.ndarray         # [32], float64
    lon: np.ndarray         # [36], float64


@dataclass(frozen=True)
class TimeSplit:
    """Contiguous time-axis split indices (no date/year literals)."""

    train: slice
    val: slice
    test: slice


def load_dataset(path: str | Path) -> ClimateDataset:
    """Load the 6 dynamic channels in contract order and the 2 static channels."""
    nc_path = Path(path)
    if not nc_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {nc_path}")

    with xr.open_dataset(nc_path) as ds:
        if "channel" in ds["data"].coords:
            da = ds["data"].sel(channel=list(CHANNELS))
        else:
            da = ds["data"]
        data = np.asarray(da.values, dtype=np.float32)

        if data.ndim != 4 or data.shape[1:] != (len(CHANNELS), NLAT, NLON):
            raise ValueError(f"Unexpected data shape {data.shape}, expected (T, {len(CHANNELS)}, {NLAT}, {NLON})")

        if "static" not in ds:
            raise KeyError("Dataset missing 'static' variable")
        st = ds["static"]
        if "static_channel" in st.coords:
            st = st.sel(static_channel=list(STATIC_CHANNELS))
        static = np.asarray(st.values, dtype=np.float32)
        if static.shape != (len(STATIC_CHANNELS), NLAT, NLON):
            raise ValueError(f"Unexpected static shape {static.shape}, expected (2, {NLAT}, {NLON})")

        doy = np.asarray(ds["time"].dt.dayofyear.values, dtype=np.int32)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)

    if lat[0] > lat[-1]:
        raise ValueError("Expected latitude ordered south-to-north (row 0 = south)")

    return ClimateDataset(data=data, static=static, doy=doy, lat=lat, lon=lon)


def compute_time_split(
    n_time: int,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
) -> TimeSplit:
    """Split [0, n_time) by fractions of the time axis into train, val, and final check."""
    if n_time <= 0:
        raise ValueError(f"n_time must be positive, got {n_time}")
    if not (0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0 and train_frac + val_frac < 1.0):
        raise ValueError(f"Invalid split fractions: train={train_frac}, val={val_frac}")

    n_train = int(round(n_time * train_frac))
    n_val = int(round(n_time * val_frac))
    n_test = n_time - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"Split produced empty partition for n_time={n_time}")

    return TimeSplit(
        train=slice(0, n_train),
        val=slice(n_train, n_train + n_val),
        test=slice(n_train + n_val, n_time),
    )


def build_windows(
    data: np.ndarray,
    doy: np.ndarray,
    t_in: int = T_IN,
    horizon: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows of t_in consecutive input days and horizon target days.

    Returns:
        init_states: [N, t_in, 6, 32, 36]
        targets:     [N, horizon, 6, 32, 36]
        init_doy_sc: [N, 2] sin/cos of the LAST input day
        target_doy:  [N, horizon] day-of-year for each target step
    """
    n_time = data.shape[0]
    total = t_in + horizon
    n_win = n_time - total + 1
    if n_win <= 0:
        raise ValueError(f"Time length {n_time} too short for t_in={t_in} + horizon={horizon}")

    init_states = np.stack([data[i : i + t_in] for i in range(n_win)], axis=0)
    targets = np.stack([data[i + t_in : i + total] for i in range(n_win)], axis=0)

    last_doy = doy[t_in - 1 : t_in - 1 + n_win].astype(np.float32)
    angle = 2.0 * np.pi * last_doy / 365.0
    init_doy_sc = np.stack([np.sin(angle), np.cos(angle)], axis=-1).astype(np.float32)

    target_doy = np.stack([doy[i + t_in : i + total] for i in range(n_win)], axis=0)
    return init_states, targets, init_doy_sc, target_doy
