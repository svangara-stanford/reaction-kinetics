"""
Descriptive metrics: per-particle (mean SOC, std, onset_soc, onset_rate, valid coverage),
persistent fast/slow, boundary vs interior, global summaries.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from reaction_kinetics.config import (
    ONSET_MEAN_ABSRATE_FRACTION_OF_MAX,
    ONSET_MEAN_ABSRATE_THRESHOLD,
    ONSET_SOC_DELTA_THRESHOLD,
)
from reaction_kinetics.schema import DescriptiveSummary, ParticleEpisode, RateMaps


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
    n_valid_rate_t = np.zeros(T_rate)
    for t in range(T_rate):
        v = valid_rate[t]
        if np.any(v):
            vals = np.abs(dc_dt[t][v])
            vals = vals[np.isfinite(vals)]
            if len(vals):
                mean_abs_rate_t[t] = np.mean(vals)
            n_valid_rate_t[t] = np.sum(v & np.isfinite(dc_dt[t]))
    fraction_valid_dc_dt_over_union_t = n_valid_rate_t / np.maximum(1, area_ref)
    frac_valid_dc_dt_t = fraction_valid_dc_dt_over_union_t  # alias
    if overlap_area_t is not None:
        fraction_valid_dc_dt_within_active_pair_t = n_valid_rate_t / np.maximum(1, overlap_area_t)
    else:
        fraction_valid_dc_dt_within_active_pair_t = np.full(T_rate, np.nan)

    union_area = area_ref
    mean_mask_area = float(np.nanmean(mask_area_t)) if mask_area_t is not None else np.nan

    # Onset: SOC-threshold-based
    c0 = mean_soc_t[0] if np.isfinite(mean_soc_t[0]) else np.nanmean(mean_soc_t)
    onset_soc_timestep: Optional[int] = None
    for t in range(1, T):
        if np.isfinite(mean_soc_t[t]) and np.abs(mean_soc_t[t] - c0) >= ONSET_SOC_DELTA_THRESHOLD:
            onset_soc_timestep = t
            break

    # Onset: mean-|dc/dt|-threshold-based (data-adaptive: fraction of max mean |dc/dt|)
    onset_rate_timestep: Optional[int] = None
    max_rate = np.nanmax(mean_abs_rate_t)
    if np.isfinite(max_rate) and max_rate > 0:
        thresh = max(ONSET_MEAN_ABSRATE_THRESHOLD, ONSET_MEAN_ABSRATE_FRACTION_OF_MAX * max_rate)
        for t in range(T_rate):
            if np.isfinite(mean_abs_rate_t[t]) and mean_abs_rate_t[t] >= thresh:
                onset_rate_timestep = t
                break

    peak_rate_t = int(np.nanargmax(mean_abs_rate_t)) if np.any(np.isfinite(mean_abs_rate_t)) else None

    # Normalized rate map (persistent fast/slow): mean over time of dc_dt / mean_particle_rate(t)
    mean_abs_rate_per_t = np.nanmax(mean_abs_rate_t)  # or per-t scale
    if np.isfinite(mean_abs_rate_per_t) and mean_abs_rate_per_t > 0:
        scale = np.maximum(mean_abs_rate_t.reshape(-1, 1, 1), 1e-12)
        norm_rate = np.where(valid_rate, dc_dt / scale, np.nan)
        with np.errstate(invalid="ignore"):
            persistent_norm_rate = np.nanmean(norm_rate, axis=0)
        if not np.any(np.isfinite(persistent_norm_rate)):
            persistent_norm_rate = np.full(dc_dt.shape[1:], np.nan)
    else:
        persistent_norm_rate = np.full(dc_dt.shape[1:], np.nan)

    return {
        "particle_id": episode.particle_id,
        "mean_soc_t": mean_soc_t,
        "std_soc_t": std_soc_t,
        "var_soc_t": var_soc_t,
        "mean_abs_rate_t": mean_abs_rate_t,
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
        "onset_soc_timestep": onset_soc_timestep,
        "onset_rate_timestep": onset_rate_timestep,
        "peak_rate_timestep": peak_rate_t,
        "persistent_norm_rate_yx": persistent_norm_rate,
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
