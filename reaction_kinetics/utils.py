"""Small helpers: safe divide, shape checks, timestep duration."""

from typing import Tuple

import numpy as np


def safe_divide(
    num: np.ndarray, denom: np.ndarray, out: np.ndarray | None = None
) -> np.ndarray:
    """Return num/denom with NaN where denom is zero. Does not fabricate zeros."""
    if out is None:
        out = np.full_like(num, np.nan, dtype=float)
    else:
        out[:] = np.nan
    valid = denom != 0
    np.divide(num, denom, where=valid, out=out)
    return out


def timestep_durations_s(
    start_time_h: np.ndarray, end_time_h: np.ndarray
) -> np.ndarray:
    """Duration in seconds for each timestep. Length T."""
    h_to_s = 3600.0
    return (end_time_h - start_time_h) * h_to_s


def interval_dt_s(
    start_time_h: np.ndarray, end_time_h: np.ndarray
) -> np.ndarray:
    """Δt in seconds for each interval between consecutive timesteps. Length T-1."""
    # dt[i] = (t[i+1] - t[i]) approx using midpoints: use end[i] and start[i+1] or average
    # Simpler: dt[i] = (start[i+1] - start[i]) or (end[i+1] - end[i])
    h_to_s = 3600.0
    start_s = start_time_h * h_to_s
    return np.diff(start_s)


def check_grid_shape(
    x: np.ndarray, y: np.ndarray, expected_nx: int, expected_ny: int
) -> Tuple[int, int]:
    """Validate that (x,y) span expected_nx * expected_ny and return (ny, nx)."""
    uniq_x = np.unique(x)
    uniq_y = np.unique(y)
    if len(uniq_x) != expected_nx or len(uniq_y) != expected_ny:
        raise ValueError(
            f"Expected grid {expected_nx}x{expected_ny}, got "
            f"x in {len(uniq_x)} values, y in {len(uniq_y)} values"
        )
    if len(x) != expected_nx * expected_ny:
        raise ValueError(
            f"Expected {expected_nx * expected_ny} rows, got {len(x)}"
        )
    return expected_ny, expected_nx
