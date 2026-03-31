"""
Descriptive metrics: per-particle (mean SOC, std, onset_soc, onset_rate, valid coverage),
persistent fast/slow, boundary vs interior, global summaries.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from reaction_kinetics.config import (
    INTERIOR_THRESHOLD_PX,
    N_RATE_PERSISTENCE_WINDOWS,
    ONSET_RATE_BASELINE_METHOD,
    ONSET_RATE_BASELINE_N_FIRST,
    ONSET_RATE_DELTA_FRACTION,
    ONSET_RATE_SMOOTH_WINDOW,
    ONSET_SOC_DELTA_THRESHOLD,
)
from reaction_kinetics.schema import DescriptiveSummary, ParticleEpisode, RateMaps


def get_boundary_interior_masks_crop(
    episode: ParticleEpisode,
    interior_threshold_px: float = INTERIOR_THRESHOLD_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (boundary_crop_xy, interior_crop_xy) both (Hy, Wx) bool in crop space."""
    x0, y0 = episode.crop_origin_xy
    Hy, Wx = episode.soc_movie_tyx.shape[1], episode.soc_movie_tyx.shape[2]
    ge = episode.geometry
    boundary_crop = ge.boundary_xy[y0 : y0 + Hy, x0 : x0 + Wx]
    dist_crop = ge.distance_to_boundary_px[y0 : y0 + Hy, x0 : x0 + Wx]
    mask_union_crop = ge.mask_xy[y0 : y0 + Hy, x0 : x0 + Wx]
    interior_crop = (dist_crop >= interior_threshold_px) & mask_union_crop
    return boundary_crop, interior_crop


def _nanmean(arr: np.ndarray, axis: Optional[tuple] = None) -> np.ndarray:
    out = np.nanmean(arr, axis=axis)
    return np.atleast_1d(out) if np.isscalar(out) else out


def _nanstd(arr: np.ndarray, axis: Optional[tuple] = None) -> np.ndarray:
    out = np.nanstd(arr, axis=axis)
    return np.atleast_1d(out) if np.isscalar(out) else out


def per_particle_metrics(
    episode: ParticleEpisode,
    rate_maps: RateMaps,
) -> Dict[str, Any]:
    """
    Per-particle: mean SOC vs t, std, var, mean |dc/dt| vs t, fraction valid SOC,
    fraction valid dc/dt, onset_soc (SOC-threshold), onset_rate (mean-|dc/dt|-threshold),
    peak rate time, lag vs global (if global_mean_soc provided).
    """
    soc = episode.soc_movie_tyx  # (T, Y, X)
    valid = episode.valid_mask_tyx
    T = soc.shape[0]
    # Masked means over spatial (valid pixels only)
    mean_soc_t = np.full(T, np.nan)
    std_soc_t = np.full(T, np.nan)
    var_soc_t = np.full(T, np.nan)
    n_valid_soc_t = np.zeros(T)
    n_total_t = np.zeros(T)
    area_ref = episode.geometry.area_px  # reference = union
    for t in range(T):
        v = valid[t]
        if np.any(v):
            vals = soc[t][v]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                mean_soc_t[t] = np.mean(vals)
                std_soc_t[t] = np.std(vals)
                var_soc_t[t] = np.var(vals)
            n_valid_soc_t[t] = np.sum(v & np.isfinite(soc[t]))
        else:
            n_valid_soc_t[t] = 0
    fraction_valid_soc_over_union_t = n_valid_soc_t / np.maximum(1, area_ref)
    frac_valid_soc_t = fraction_valid_soc_over_union_t  # alias

    mask_area_t = episode.metadata.get("mask_area_t")
    overlap_area_t = episode.metadata.get("overlap_area_t")
    if mask_area_t is not None:
        fraction_valid_soc_within_mask_t = n_valid_soc_t / np.maximum(1, mask_area_t)
        mask_area_t_over_union_t = mask_area_t / np.maximum(1, area_ref)
        mean_occupancy_fraction = float(np.nanmean(mask_area_t_over_union_t))
        valid_t = mask_area_t > 0
        mean_conditional_soc_validity = (
            float(np.nanmean(fraction_valid_soc_within_mask_t[valid_t]))
            if np.any(valid_t) else np.nan
        )
    else:
        fraction_valid_soc_within_mask_t = np.full(T, np.nan)
        mask_area_t_over_union_t = np.full(T, np.nan)
        mean_occupancy_fraction = np.nan
        mean_conditional_soc_validity = np.nan

    # Rate: T-1
    dc_dt = rate_maps.dc_dt_tyx
    valid_rate = rate_maps.valid_mask_tyx
    T_rate = dc_dt.shape[0]
    mean_abs_rate_t = np.full(T_rate, np.nan)
    mean_dc_dt_t = np.full(T_rate, np.nan)
    n_valid_rate_t = np.zeros(T_rate)
    for t in range(T_rate):
        v = valid_rate[t]
        if np.any(v):
            vals_abs = np.abs(dc_dt[t][v])
            vals_abs = vals_abs[np.isfinite(vals_abs)]
            if len(vals_abs):
                mean_abs_rate_t[t] = np.mean(vals_abs)
            vals_signed = dc_dt[t][v]
            vals_signed = vals_signed[np.isfinite(vals_signed)]
            if len(vals_signed):
                mean_dc_dt_t[t] = np.mean(vals_signed)
            n_valid_rate_t[t] = np.sum(v & np.isfinite(dc_dt[t]))
    fraction_valid_dc_dt_over_union_t = n_valid_rate_t / np.maximum(1, area_ref)
    mean_conditional_rate_validity = float(np.nanmean(fraction_valid_dc_dt_over_union_t))
    frac_valid_dc_dt_t = fraction_valid_dc_dt_over_union_t  # alias
    if overlap_area_t is not None:
        fraction_valid_dc_dt_within_active_pair_t = n_valid_rate_t / np.maximum(1, overlap_area_t)
    else:
        fraction_valid_dc_dt_within_active_pair_t = np.full(T_rate, np.nan)

    union_area = area_ref
    mean_mask_area = float(np.nanmean(mask_area_t)) if mask_area_t is not None else np.nan

    # Boundary vs interior (crop-space masks)
    boundary_crop, interior_crop = get_boundary_interior_masks_crop(episode)
    mean_soc_boundary_t = np.full(T, np.nan)
    mean_soc_interior_t = np.full(T, np.nan)
    for t in range(T):
        vb = valid[t] & boundary_crop
        vi = valid[t] & interior_crop
        if np.any(vb):
            vals = soc[t][vb]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                mean_soc_boundary_t[t] = np.mean(vals)
        if np.any(vi):
            vals = soc[t][vi]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                mean_soc_interior_t[t] = np.mean(vals)
    mean_abs_rate_boundary_t = np.full(T_rate, np.nan)
    mean_abs_rate_interior_t = np.full(T_rate, np.nan)
    for t in range(T_rate):
        vb = valid_rate[t] & boundary_crop
        vi = valid_rate[t] & interior_crop
        if np.any(vb):
            vals = np.abs(dc_dt[t][vb])
            vals = vals[np.isfinite(vals)]
            if len(vals):
                mean_abs_rate_boundary_t[t] = np.mean(vals)
        if np.any(vi):
            vals = np.abs(dc_dt[t][vi])
            vals = vals[np.isfinite(vals)]
            if len(vals):
                mean_abs_rate_interior_t[t] = np.mean(vals)
    mean_soc_boundary_global = float(np.nanmean(mean_soc_boundary_t))
    mean_soc_interior_global = float(np.nanmean(mean_soc_interior_t))
    mean_abs_rate_boundary_global = float(np.nanmean(mean_abs_rate_boundary_t))
    mean_abs_rate_interior_global = float(np.nanmean(mean_abs_rate_interior_t))

    # Onset: SOC-threshold-based
    c0 = mean_soc_t[0] if np.isfinite(mean_soc_t[0]) else np.nanmean(mean_soc_t)
    onset_soc_timestep: Optional[int] = None
    for t in range(1, T):
        if np.isfinite(mean_soc_t[t]) and np.abs(mean_soc_t[t] - c0) >= ONSET_SOC_DELTA_THRESHOLD:
            onset_soc_timestep = t
            break

    # Onset rate: particle baseline (first N) + smoothed rate + delta above baseline
    onset_rate_timestep: Optional[int] = None
    n_baseline = min(ONSET_RATE_BASELINE_N_FIRST, T_rate)
    baseline_vals = mean_abs_rate_t[:n_baseline]
    baseline_vals = baseline_vals[np.isfinite(baseline_vals)]
    baseline = float(np.median(baseline_vals)) if ONSET_RATE_BASELINE_METHOD == "median" and len(baseline_vals) else (float(np.mean(baseline_vals)) if len(baseline_vals) else np.nan)
    if not np.isfinite(baseline):
        baseline = np.nanmin(mean_abs_rate_t) if np.any(np.isfinite(mean_abs_rate_t)) else np.nan
    smooth_window = min(ONSET_RATE_SMOOTH_WINDOW if ONSET_RATE_SMOOTH_WINDOW % 2 == 1 else ONSET_RATE_SMOOTH_WINDOW + 1, T_rate)
    if smooth_window >= 3 and np.sum(np.isfinite(mean_abs_rate_t)) >= smooth_window:
        mean_abs_rate_smooth_t = savgol_filter(np.nan_to_num(mean_abs_rate_t, nan=np.nanmean(mean_abs_rate_t)), smooth_window, min(2, smooth_window - 1))
    else:
        mean_abs_rate_smooth_t = mean_abs_rate_t.copy()
    max_smooth = np.nanmax(mean_abs_rate_smooth_t)
    if np.isfinite(baseline) and np.isfinite(max_smooth) and max_smooth > baseline:
        thresh = baseline + ONSET_RATE_DELTA_FRACTION * (max_smooth - baseline)
        for t in range(T_rate):
            if np.isfinite(mean_abs_rate_smooth_t[t]) and mean_abs_rate_smooth_t[t] >= thresh:
                onset_rate_timestep = t
                break

    peak_rate_t = int(np.nanargmax(mean_abs_rate_t)) if np.any(np.isfinite(mean_abs_rate_t)) else None

    # Normalized rate: per-t and time-aggregated persistent maps.
    # Formula:
    #   Input: signed dc_dt (T-1, Y, X); mean_abs_rate_t[t] = mean over valid pixels of |dc_dt[t]|.
    #   norm_rate_tyx[t] = dc_dt[t] / mean_abs_rate_t[t] (NaN where invalid). Units: multiples of
    #   particle mean |dc/dt| at that t; positive = faster than particle average, negative = slower.
    #   persistent_norm_rate_yx = nanmean(norm_rate_tyx, axis=0) = time-mean (zero ≈ at particle average).
    mean_abs_rate_per_t = np.nanmax(mean_abs_rate_t)
    if np.isfinite(mean_abs_rate_per_t) and mean_abs_rate_per_t > 0:
        scale = np.maximum(mean_abs_rate_t.reshape(-1, 1, 1), 1e-12)
        norm_rate_tyx = np.where(valid_rate, dc_dt / scale, np.nan)
        with np.errstate(invalid="ignore"):
            persistent_norm_rate = np.nanmean(norm_rate_tyx, axis=0)
            persistent_norm_rate_median_yx = np.nanmedian(norm_rate_tyx, axis=0)
        if not np.any(np.isfinite(persistent_norm_rate)):
            persistent_norm_rate = np.full(dc_dt.shape[1:], np.nan)
        if not np.any(np.isfinite(persistent_norm_rate_median_yx)):
            persistent_norm_rate_median_yx = np.full(dc_dt.shape[1:], np.nan)
        # Deviation from particle-mean (normalized) at each t: positive = persistently above average.
        spatial_mean_norm_t = np.nanmean(norm_rate_tyx.reshape(norm_rate_tyx.shape[0], -1), axis=1)
        deviation_norm_tyx = norm_rate_tyx - spatial_mean_norm_t.reshape(-1, 1, 1)
        with np.errstate(invalid="ignore"):
            persistent_deviation_norm_yx = np.nanmean(deviation_norm_tyx, axis=0)
        if not np.any(np.isfinite(persistent_deviation_norm_yx)):
            persistent_deviation_norm_yx = np.full(dc_dt.shape[1:], np.nan)
    else:
        norm_rate_tyx = np.full_like(dc_dt, np.nan)
        persistent_norm_rate = np.full(dc_dt.shape[1:], np.nan)
        persistent_norm_rate_median_yx = np.full(dc_dt.shape[1:], np.nan)
        persistent_deviation_norm_yx = np.full(dc_dt.shape[1:], np.nan)

    # Persistence: correlation of normalized rate maps across time windows
    K = min(N_RATE_PERSISTENCE_WINDOWS, T_rate)
    norm_rate_window_correlation_matrix = np.full((K, K), np.nan)
    if K >= 2 and np.any(np.isfinite(norm_rate_tyx)):
        window_maps = []
        for w in range(K):
            i0 = (T_rate * w) // K
            i1 = (T_rate * (w + 1)) // K
            if i1 <= i0:
                continue
            wmap = np.nanmean(norm_rate_tyx[i0:i1], axis=0)
            window_maps.append(wmap.ravel())
        # Pairwise Pearson over valid pixels
        valid_flat = np.any(np.isfinite(norm_rate_tyx), axis=0).ravel()
        for i in range(len(window_maps)):
            for j in range(len(window_maps)):
                a, b = window_maps[i], window_maps[j]
                mask = valid_flat & np.isfinite(a) & np.isfinite(b)
                if np.sum(mask) > 2:
                    norm_rate_window_correlation_matrix[i, j] = np.corrcoef(a[mask], b[mask])[0, 1]
        off_diag = []
        for i in range(K):
            for j in range(K):
                if i != j and np.isfinite(norm_rate_window_correlation_matrix[i, j]):
                    off_diag.append(norm_rate_window_correlation_matrix[i, j])
        norm_rate_persistence_mean_corr = float(np.mean(off_diag)) if off_diag else np.nan
    else:
        norm_rate_persistence_mean_corr = np.nan

    # Segment aggregates (early_rising, late_rising, early_falling, late_falling)
    segment_label_t = episode.metadata.get("segment_label_t")  # (T,)
    segment_heterogeneity: Dict[str, float] = {}
    segment_mean_abs_rate: Dict[str, float] = {}
    segment_n_timesteps: Dict[str, int] = {}
    segment_mean_norm_rate_yx: Dict[str, np.ndarray] = {}
    if segment_label_t is not None and len(segment_label_t) == T:
        segment_label_rate = segment_label_t[:T_rate]  # (T-1,) for rate intervals
        for seg_name in ("early_rising", "late_rising", "early_falling", "late_falling"):
            mask_soc = segment_label_t == seg_name
            mask_rate = segment_label_rate == seg_name
            if np.any(mask_soc):
                segment_heterogeneity[seg_name] = float(np.nanmean(std_soc_t[mask_soc]))
                segment_n_timesteps[seg_name] = int(np.sum(mask_soc))
            else:
                segment_heterogeneity[seg_name] = np.nan
                segment_n_timesteps[seg_name] = 0
            if np.any(mask_rate):
                segment_mean_abs_rate[seg_name] = float(np.nanmean(mean_abs_rate_t[mask_rate]))
                with np.errstate(invalid="ignore"):
                    segment_mean_norm_rate_yx[seg_name] = np.nanmean(norm_rate_tyx[mask_rate], axis=0)
            else:
                segment_mean_abs_rate[seg_name] = np.nan
                segment_mean_norm_rate_yx[seg_name] = np.full(dc_dt.shape[1:], np.nan)

    return {
        "particle_id": episode.particle_id,
        "mean_soc_t": mean_soc_t,
        "std_soc_t": std_soc_t,
        "var_soc_t": var_soc_t,
        "mean_abs_rate_t": mean_abs_rate_t,
        "mean_dc_dt_t": mean_dc_dt_t,
        "frac_valid_soc_t": frac_valid_soc_t,
        "frac_valid_dc_dt_t": frac_valid_dc_dt_t,
        "fraction_valid_soc_over_union_t": fraction_valid_soc_over_union_t,
        "fraction_valid_soc_within_mask_t": fraction_valid_soc_within_mask_t,
        "fraction_valid_dc_dt_over_union_t": fraction_valid_dc_dt_over_union_t,
        "fraction_valid_dc_dt_within_active_pair_t": fraction_valid_dc_dt_within_active_pair_t,
        "mask_area_t_over_union_t": mask_area_t_over_union_t,
        "union_area": union_area,
        "mean_mask_area": mean_mask_area,
        "mean_occupancy_fraction": mean_occupancy_fraction,
        "mean_conditional_soc_validity": mean_conditional_soc_validity,
        "mean_conditional_rate_validity": mean_conditional_rate_validity,
        "onset_soc_timestep": onset_soc_timestep,
        "onset_rate_timestep": onset_rate_timestep,
        "peak_rate_timestep": peak_rate_t,
        "persistent_norm_rate_yx": persistent_norm_rate,
        "persistent_norm_rate_median_yx": persistent_norm_rate_median_yx,
        "persistent_deviation_norm_yx": persistent_deviation_norm_yx,
        "norm_rate_tyx": norm_rate_tyx,
        "norm_rate_window_correlation_matrix": norm_rate_window_correlation_matrix,
        "norm_rate_persistence_mean_corr": norm_rate_persistence_mean_corr,
        "mean_soc_boundary_t": mean_soc_boundary_t,
        "mean_soc_interior_t": mean_soc_interior_t,
        "mean_abs_rate_boundary_t": mean_abs_rate_boundary_t,
        "mean_abs_rate_interior_t": mean_abs_rate_interior_t,
        "mean_soc_boundary_global": mean_soc_boundary_global,
        "mean_soc_interior_global": mean_soc_interior_global,
        "mean_abs_rate_boundary_global": mean_abs_rate_boundary_global,
        "mean_abs_rate_interior_global": mean_abs_rate_interior_global,
        "segment_heterogeneity": segment_heterogeneity,
        "segment_mean_abs_rate": segment_mean_abs_rate,
        "segment_n_timesteps": segment_n_timesteps,
        "segment_mean_norm_rate_yx": segment_mean_norm_rate_yx,
        "time_s": episode.time_s,
        "time_mid_s": rate_maps.time_mid_s,
    }


def global_metrics(
    per_particle: List[Dict[str, Any]],
    all_particle_mean_soc: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Global: all-particle mean SOC curve, onset distributions, pairwise correlations placeholder."""
    out: Dict[str, Any] = {}
    if not per_particle:
        return out
    # Stack mean_soc_t and average
    T = len(per_particle[0]["mean_soc_t"])
    stacked = np.array([p["mean_soc_t"] for p in per_particle])
    out["global_mean_soc_t"] = np.nanmean(stacked, axis=0)
    out["global_std_soc_t"] = np.nanstd(stacked, axis=0)
    out["onset_soc_timesteps"] = [p["onset_soc_timestep"] for p in per_particle]
    out["onset_rate_timesteps"] = [p["onset_rate_timestep"] for p in per_particle]
    return out


def compute_descriptive_summary(
    episodes: List[ParticleEpisode],
    rate_maps_per_particle: List[RateMaps],
    figure_paths: Optional[List[str]] = None,
    table_paths: Optional[List[str]] = None,
) -> DescriptiveSummary:
    """Aggregate per-particle metrics and global metrics."""
    if len(episodes) != len(rate_maps_per_particle):
        raise ValueError("episodes and rate_maps_per_particle must have same length")
    per_particle = []
    for ep, rate in zip(episodes, rate_maps_per_particle):
        per_particle.append(per_particle_metrics(ep, rate))
    global_met = global_metrics(per_particle)
    return DescriptiveSummary(
        per_particle=per_particle,
        global_metrics=global_met,
        figure_paths=figure_paths or [],
        table_paths=table_paths or [],
    )
