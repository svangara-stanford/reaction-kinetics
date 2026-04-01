"""
dc/dt from SOC movie. Length T-1; aligned to time midpoints.
Optional Savitzky-Golay smoothing in time; valid only where both c(t) and c(t+1) are finite.
"""

from typing import Any, Dict, Optional

import numpy as np
from scipy.signal import savgol_filter

from reaction_kinetics.schema import RateMaps


def compute_dc_dt(
    soc_tyx: np.ndarray,
    time_s: np.ndarray,
    valid_mask_tyx: Optional[np.ndarray] = None,
    smooth_window: Optional[int] = None,
    smooth_poly: int = 2,
    erode_boundary_px: int = 0,
) -> RateMaps:
    """
    Compute dc/dt at midpoints. dc_dt_tyx has shape (T-1, Y, X).
    time_mid_s[i] = (time_s[i] + time_s[i+1]) / 2.
    dt_s[i] = time_s[i+1] - time_s[i].
    Valid only where both c(t) and c(t+1) are finite; do not blur across NaN.
    """
    T, ny, nx = soc_tyx.shape
    if len(time_s) != T:
        raise ValueError(f"time_s length {len(time_s)} != T {T}")

    time_mid_s = (time_s[:-1] + time_s[1:]) / 2.0
    dt_s = np.diff(time_s)
    # dt_s is (T-1,); broadcast to (T-1, 1, 1) for division
    dt_bc = dt_s.reshape(-1, 1, 1)

    c = soc_tyx.astype(float)
    if smooth_window is not None and smooth_window >= 3 and T >= smooth_window:
        # Savitzky-Golay in time per pixel; only where full series is finite (no fabricating across NaN)
        c_smooth = c.copy()
        for y in range(ny):
            for x in range(nx):
                series = c[:, y, x]
                if np.all(np.isfinite(series)) and len(series) >= smooth_window:
                    try:
                        c_smooth[:, y, x] = savgol_filter(series, smooth_window, smooth_poly)
                    except Exception:
                        pass
        c = c_smooth

    dc_dt_tyx = (c[1:] - c[:-1]) / dt_bc
    # Valid only where both adjacent SOC values are finite
    both_finite = np.isfinite(c[:-1]) & np.isfinite(c[1:])
    valid_mask_tyx_out = both_finite.copy()

    if erode_boundary_px > 0:
        from scipy import ndimage
        struct = np.ones((2 * erode_boundary_px + 1, 2 * erode_boundary_px + 1))
        for t in range(valid_mask_tyx_out.shape[0]):
            valid_mask_tyx_out[t] = ndimage.binary_erosion(
                valid_mask_tyx_out[t], structure=struct
            )

    if valid_mask_tyx is not None:
        # Intersect with provided mask (e.g. particle mask per timestep)
        if valid_mask_tyx.shape[0] == T - 1:
            valid_mask_tyx_out = valid_mask_tyx_out & valid_mask_tyx
        elif valid_mask_tyx.shape[0] == T:
            valid_mask_tyx_out = valid_mask_tyx_out & valid_mask_tyx[:-1]

    # Where not valid, set dc/dt to NaN
    dc_dt_tyx = np.where(valid_mask_tyx_out, dc_dt_tyx, np.nan)

    meta: Dict[str, Any] = {
        "smooth_window": smooth_window,
        "smooth_poly": smooth_poly,
        "erode_boundary_px": erode_boundary_px,
    }
    return RateMaps(
        dc_dt_tyx=dc_dt_tyx,
        time_mid_s=time_mid_s,
        valid_mask_tyx=valid_mask_tyx_out,
        dt_s=dt_s,
        smoothing_metadata=meta,
    )


def compute_dx_li_dt(
    x_li_tyx: np.ndarray,
    time_s: np.ndarray,
    valid_mask_tyx: Optional[np.ndarray] = None,
    smooth_window: Optional[int] = None,
    smooth_poly: int = 2,
    erode_boundary_px: int = 0,
) -> RateMaps:
    """Alias of compute_dc_dt for x_li(t,y,x), preserving T-1 midpoint alignment."""
    return compute_dc_dt(
        x_li_tyx,
        time_s,
        valid_mask_tyx=valid_mask_tyx,
        smooth_window=smooth_window,
        smooth_poly=smooth_poly,
        erode_boundary_px=erode_boundary_px,
    )
