"""
Boundary-only kinetics: collect boundary (c, dc/dt), pooled normalized data,
binned fits on common SOC grid, full-particle comparison, R².
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from reaction_kinetics.config import BOUNDARY_FIT_SOC_GRID_SIZE
from reaction_kinetics.descriptive import get_boundary_interior_masks_crop
from reaction_kinetics.schema import ParticleEpisode, RateMaps


def collect_boundary_only_kinetics(
    episodes: List[ParticleEpisode],
    rate_maps_list: List[RateMaps],
    exclude_particle_ids: List[int],
) -> Tuple[
    Dict[int, Dict[str, Any]],
    Dict[str, Tuple[np.ndarray, np.ndarray]],
    Dict[str, Tuple[np.ndarray, np.ndarray]],
]:
    """
    Collect (SOC_mid, dc/dt) at boundary pixels only, for retained particles.
    Returns (per_particle, pooled_normalized, pooled_raw).
    pooled_normalized: for plots (dc normalized by particle-segment mean |dc/dt|).
    pooled_raw: for fitting and comparison with full-particle (same units).
    """
    per_particle: Dict[int, Dict[str, Any]] = {}
    pool_rise_c: List[np.ndarray] = []
    pool_rise_dc: List[np.ndarray] = []
    pool_fall_c: List[np.ndarray] = []
    pool_fall_dc: List[np.ndarray] = []
    pool_rise_c_raw: List[np.ndarray] = []
    pool_rise_dc_raw: List[np.ndarray] = []
    pool_fall_c_raw: List[np.ndarray] = []
    pool_fall_dc_raw: List[np.ndarray] = []

    for ep, rate in zip(episodes, rate_maps_list):
        if ep.particle_id in exclude_particle_ids:
            continue
        boundary_crop, _ = get_boundary_interior_masks_crop(ep)
        soc = ep.soc_movie_tyx
        dc_dt = rate.dc_dt_tyx
        valid_rate = rate.valid_mask_tyx
        T = soc.shape[0]
        c_rise, dc_rise = [], []
        c_fall, dc_fall = [], []
        for t in range(T - 1):
            c_mid = (soc[t] + soc[t + 1]) / 2
            mask = (
                valid_rate[t]
                & boundary_crop
                & np.isfinite(dc_dt[t])
                & np.isfinite(c_mid)
            )
            if not np.any(mask):
                continue
            c_vals = c_mid[mask].ravel()
            dc_vals = dc_dt[t][mask].ravel()
            if ep.drive_sign[t] > 0:
                c_rise.extend(c_vals)
                dc_rise.extend(dc_vals)
            elif ep.drive_sign[t] < 0:
                c_fall.extend(c_vals)
                dc_fall.extend(dc_vals)

        c_rise = np.array(c_rise) if c_rise else np.array([])
        dc_rise = np.array(dc_rise) if dc_rise else np.array([])
        c_fall = np.array(c_fall) if c_fall else np.array([])
        dc_fall = np.array(dc_fall) if dc_fall else np.array([])

        mean_abs_rise = float(np.mean(np.abs(dc_rise))) if len(dc_rise) else np.nan
        mean_abs_fall = float(np.mean(np.abs(dc_fall))) if len(dc_fall) else np.nan
        scale_rise = mean_abs_rise if np.isfinite(mean_abs_rise) and mean_abs_rise > 0 else 1.0
        scale_fall = mean_abs_fall if np.isfinite(mean_abs_fall) and mean_abs_fall > 0 else 1.0

        per_particle[ep.particle_id] = {
            "rising": (c_rise, dc_rise),
            "falling": (c_fall, dc_fall),
            "mean_abs_rate_rising": mean_abs_rise,
            "mean_abs_rate_falling": mean_abs_fall,
            "n_rising": len(c_rise),
            "n_falling": len(c_fall),
        }
        if len(c_rise):
            pool_rise_c.append(c_rise)
            pool_rise_dc.append(dc_rise / scale_rise)
            pool_rise_c_raw.append(c_rise)
            pool_rise_dc_raw.append(dc_rise)
        if len(c_fall):
            pool_fall_c.append(c_fall)
            pool_fall_dc.append(dc_fall / scale_fall)
            pool_fall_c_raw.append(c_fall)
            pool_fall_dc_raw.append(dc_fall)

    pooled_normalized = {
        "rising": (
            np.concatenate(pool_rise_c) if pool_rise_c else np.array([]),
            np.concatenate(pool_rise_dc) if pool_rise_dc else np.array([]),
        ),
        "falling": (
            np.concatenate(pool_fall_c) if pool_fall_c else np.array([]),
            np.concatenate(pool_fall_dc) if pool_fall_dc else np.array([]),
        ),
    }
    pooled_raw = {
        "rising": (
            np.concatenate(pool_rise_c_raw) if pool_rise_c_raw else np.array([]),
            np.concatenate(pool_rise_dc_raw) if pool_rise_dc_raw else np.array([]),
        ),
        "falling": (
            np.concatenate(pool_fall_c_raw) if pool_fall_c_raw else np.array([]),
            np.concatenate(pool_fall_dc_raw) if pool_fall_dc_raw else np.array([]),
        ),
    }
    return per_particle, pooled_normalized, pooled_raw


def collect_full_particle_pooled(
    episodes: List[ParticleEpisode],
    rate_maps_list: List[RateMaps],
    exclude_particle_ids: List[int],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Pool (c, dc/dt) from all pixels for retained particles, by segment. No normalization."""
    pool_rise_c: List[np.ndarray] = []
    pool_rise_dc: List[np.ndarray] = []
    pool_fall_c: List[np.ndarray] = []
    pool_fall_dc: List[np.ndarray] = []

    for ep, rate in zip(episodes, rate_maps_list):
        if ep.particle_id in exclude_particle_ids:
            continue
        soc = ep.soc_movie_tyx
        dc_dt = rate.dc_dt_tyx
        valid_rate = rate.valid_mask_tyx
        T = soc.shape[0]
        for t in range(T - 1):
            c_mid = (soc[t] + soc[t + 1]) / 2
            mask = valid_rate[t] & np.isfinite(dc_dt[t]) & np.isfinite(c_mid)
            if not np.any(mask):
                continue
            c_vals = c_mid[mask].ravel()
            dc_vals = dc_dt[t][mask].ravel()
            if ep.drive_sign[t] > 0:
                pool_rise_c.append(c_vals)
                pool_rise_dc.append(dc_vals)
            elif ep.drive_sign[t] < 0:
                pool_fall_c.append(c_vals)
                pool_fall_dc.append(dc_vals)

    return {
        "rising": (
            np.concatenate(pool_rise_c) if pool_rise_c else np.array([]),
            np.concatenate(pool_rise_dc) if pool_rise_dc else np.array([]),
        ),
        "falling": (
            np.concatenate(pool_fall_c) if pool_fall_c else np.array([]),
            np.concatenate(pool_fall_dc) if pool_fall_dc else np.array([]),
        ),
    }


def binned_fit_on_grid(
    c: np.ndarray,
    y: np.ndarray,
    soc_grid: np.ndarray,
    use_median: bool = True,
) -> np.ndarray:
    """Bin (c, y) by soc_grid edges; compute median (or mean) per bin; return fitted values on soc_grid (bin center or nearest)."""
    if len(c) == 0 or len(y) == 0:
        return np.full_like(soc_grid, np.nan)
    edges = np.linspace(0, 1, len(soc_grid) + 1)
    dig = np.digitize(c, edges)
    fitted = np.full(len(soc_grid), np.nan)
    for i in range(1, len(edges)):
        sel = dig == i
        if np.sum(sel) > 0:
            vals = y[sel]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                fitted[i - 1] = float(np.nanmedian(vals) if use_median else np.nanmean(vals))
    # forward-fill then backward-fill NaNs at edges
    for j in range(len(fitted)):
        if np.isnan(fitted[j]) and j > 0:
            fitted[j] = fitted[j - 1]
    for j in range(len(fitted) - 1, -1, -1):
        if np.isnan(fitted[j]) and j < len(fitted) - 1:
            fitted[j] = fitted[j + 1]
    return fitted


def predict_at_soc(fitted_values: np.ndarray, soc_grid: np.ndarray, soc_obs: np.ndarray) -> np.ndarray:
    """Linear interpolation of fitted_values (at soc_grid) to soc_obs."""
    if len(soc_grid) == 0 or len(fitted_values) == 0 or len(soc_obs) == 0:
        return np.full_like(soc_obs, np.nan)
    valid = np.isfinite(fitted_values)
    if not np.any(valid):
        return np.full_like(soc_obs, np.nan)
    return np.interp(soc_obs, soc_grid, np.nan_to_num(fitted_values, nan=np.nanmean(fitted_values[valid])))


def variance_explained(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 - SS_res / SS_tot. Returns nan if insufficient data."""
    mask = np.isfinite(y_obs) & np.isfinite(y_pred)
    if np.sum(mask) < 2:
        return np.nan
    y_o = y_obs[mask]
    y_p = y_pred[mask]
    ss_tot = np.sum((y_o - np.mean(y_o)) ** 2)
    if ss_tot == 0:
        return np.nan
    ss_res = np.sum((y_o - y_p) ** 2)
    return float(1.0 - ss_res / ss_tot)


def run_boundary_fits(
    soc_grid: np.ndarray,
    boundary_pooled: Dict[str, Tuple[np.ndarray, np.ndarray]],
    full_pooled: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, float]],
]:
    """
    Compute binned fits for boundary and full-particle data on soc_grid.
    Returns (boundary_fits, full_fits, fit_quality) where
    boundary_fits[segment][rate_type] = array len(soc_grid),
    fit_quality[segment][rate_type] = R².
    """
    boundary_fits: Dict[str, Dict[str, np.ndarray]] = {"rising": {}, "falling": {}}
    full_fits: Dict[str, Dict[str, np.ndarray]] = {"rising": {}, "falling": {}}
    fit_quality: Dict[str, Dict[str, float]] = {"rising": {}, "falling": {}}

    for seg in ("rising", "falling"):
        c_b, dc_b = boundary_pooled[seg]
        c_f, dc_f = full_pooled[seg]
        for rate_type, use_abs in [("signed", False), ("abs", True)]:
            y_b = np.abs(dc_b) if use_abs else dc_b
            y_f = np.abs(dc_f) if use_abs else dc_f
            fit_b = binned_fit_on_grid(c_b, y_b, soc_grid, use_median=True)
            fit_f = binned_fit_on_grid(c_f, y_f, soc_grid, use_median=True)
            boundary_fits[seg][rate_type] = fit_b
            full_fits[seg][rate_type] = fit_f
            pred_b = predict_at_soc(fit_b, soc_grid, c_b)
            pred_f = predict_at_soc(fit_f, soc_grid, c_f)
            fit_quality[seg][rate_type] = variance_explained(y_b, pred_b)
    return boundary_fits, full_fits, fit_quality
