"""
Diagnostic plots. dc/dt has length T-1 and is aligned to time_mid_s (midpoints).
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reaction_kinetics.schema import DescriptiveSummary, ParticleEpisode, RateMaps


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_particle_soc_curves(
    episodes: List[ParticleEpisode],
    summary: DescriptiveSummary,
    out_dir: str,
) -> str:
    """Overlay particle-average SOC vs time for all particles."""
    fig, ax = plt.subplots()
    for ep in episodes:
        mean_soc = []
        for t in range(ep.soc_movie_tyx.shape[0]):
            v = ep.soc_movie_tyx[t][ep.valid_mask_tyx[t]]
            v = v[np.isfinite(v)]
            mean_soc.append(np.mean(v) if len(v) else np.nan)
        ax.plot(ep.time_s / 3600, mean_soc, label=f"P{ep.particle_id}", alpha=0.8)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Mean SOC")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Particle-average SOC vs time")
    out_path = _ensure_dir(out_dir) / "particle_soc_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def plot_dc_dt_vs_c(
    episodes: List[ParticleEpisode],
    rate_maps_list: List[RateMaps],
    out_dir: str,
) -> str:
    """Scatter dc/dt vs c split by rising vs falling voltage (T-1 midpoints)."""
    c_rising, dc_rising = [], []
    c_falling, dc_falling = [], []
    for ep, rate in zip(episodes, rate_maps_list):
        T = ep.soc_movie_tyx.shape[0]
        c_mid = (ep.soc_movie_tyx[:-1] + ep.soc_movie_tyx[1:]) / 2
        for t in range(T - 1):
            v = rate.valid_mask_tyx[t] & np.isfinite(rate.dc_dt_tyx[t]) & np.isfinite(c_mid[t])
            if not np.any(v):
                continue
            c_vals = c_mid[t][v].ravel()
            dc_vals = rate.dc_dt_tyx[t][v].ravel()
            if ep.drive_sign[t] > 0:
                c_rising.extend(c_vals)
                dc_rising.extend(dc_vals)
            elif ep.drive_sign[t] < 0:
                c_falling.extend(c_vals)
                dc_falling.extend(dc_vals)
    fig, (ax_rise, ax_fall) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    for ax, all_c, all_dc, title in [
        (ax_rise, np.array(c_rising), np.array(dc_rising), "Rising voltage (charge)"),
        (ax_fall, np.array(c_falling), np.array(dc_falling), "Falling voltage (discharge)"),
    ]:
        if len(all_c) == 0:
            ax.set_title(title)
            ax.set_xlabel("SOC (midpoint)")
            ax.set_ylabel("dc/dt")
            ax.axhline(0, color="gray", ls="--")
            continue
        if len(all_c) > 1000:
            idx = np.random.choice(len(all_c), 1000, replace=False)
            ax.scatter(all_c[idx], all_dc[idx], alpha=0.2, s=5)
        else:
            ax.scatter(all_c, all_dc, alpha=0.3, s=5)
        if len(all_c) > 10:
            bins = np.linspace(0, 1, 11)
            dig = np.digitize(all_c, bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_means = [np.nanmean(all_dc[dig == i]) for i in range(1, len(bins))]
            valid = [np.isfinite(m) for m in bin_means]
            if any(valid):
                ax.plot(bin_centers, bin_means, "k-", lw=2, label="Binned mean")
        ax.set_xlabel("SOC (midpoint)")
        ax.set_ylabel("dc/dt")
        ax.set_title(title)
        ax.axhline(0, color="gray", ls="--")
    out_path = _ensure_dir(out_dir) / "dc_dt_vs_c.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def plot_heterogeneity_vs_time(
    summary: DescriptiveSummary,
    out_dir: str,
) -> str:
    """Within-particle std SOC vs time (and global mean)."""
    fig, ax = plt.subplots()
    for p in summary.per_particle:
        t_s = p["time_s"]
        std_soc = p["std_soc_t"]
        ax.plot(t_s / 3600, std_soc, alpha=0.7, label=f"P{p['particle_id']}")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Std SOC within particle")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Heterogeneity (std SOC) vs time")
    out_path = _ensure_dir(out_dir) / "heterogeneity_vs_time.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def _get_series(p: dict, key_primary: str, key_fallback: str):
    return p.get(key_primary) if p.get(key_primary) is not None else p.get(key_fallback)


def plot_valid_coverage(
    summary: DescriptiveSummary,
    out_dir: str,
) -> str:
    """Union-referenced and conditional valid coverage; mask area / union vs time."""
    fig, axes = plt.subplots(3, 2, sharex="col", figsize=(10, 9))
    ax_soc_union, ax_soc_cond = axes[0, 0], axes[0, 1]
    ax_dc_union, ax_dc_cond = axes[1, 0], axes[1, 1]
    ax_occ1, ax_occ2 = axes[2, 0], axes[2, 1]
    for p in summary.per_particle:
        t_s = p["time_s"]
        t_mid = p["time_mid_s"]
        soc_over_union = _get_series(p, "fraction_valid_soc_over_union_t", "frac_valid_soc_t")
        soc_within_mask = p.get("fraction_valid_soc_within_mask_t")
        dc_over_union = _get_series(p, "fraction_valid_dc_dt_over_union_t", "frac_valid_dc_dt_t")
        dc_within_pair = p.get("fraction_valid_dc_dt_within_active_pair_t")
        occ = p.get("mask_area_t_over_union_t")
        ax_soc_union.plot(t_s / 3600, soc_over_union, alpha=0.7, label=f"P{p['particle_id']}")
        if soc_within_mask is not None and np.any(np.isfinite(soc_within_mask)):
            ax_soc_cond.plot(t_s / 3600, soc_within_mask, alpha=0.7, label=f"P{p['particle_id']}")
        ax_dc_union.plot(t_mid / 3600, dc_over_union, alpha=0.7, label=f"P{p['particle_id']}")
        if dc_within_pair is not None and np.any(np.isfinite(dc_within_pair)):
            ax_dc_cond.plot(t_mid / 3600, dc_within_pair, alpha=0.7, label=f"P{p['particle_id']}")
        if occ is not None and np.any(np.isfinite(occ)):
            ax_occ1.plot(t_s / 3600, occ, alpha=0.7, label=f"P{p['particle_id']}")
    ax_soc_union.set_ylabel("Frac valid SOC")
    ax_soc_union.legend(loc="best", fontsize=7)
    ax_soc_union.set_title("SOC coverage (over union)")
    ax_soc_cond.set_ylabel("Frac valid SOC")
    ax_soc_cond.legend(loc="best", fontsize=7)
    ax_soc_cond.set_title("SOC coverage (within mask at t)")
    ax_dc_union.set_ylabel("Frac valid dc/dt")
    ax_dc_union.legend(loc="best", fontsize=7)
    ax_dc_union.set_title("dc/dt coverage (over union)")
    ax_dc_cond.set_ylabel("Frac valid dc/dt")
    ax_dc_cond.legend(loc="best", fontsize=7)
    ax_dc_cond.set_title("dc/dt coverage (within active pair)")
    ax_occ1.set_xlabel("Time (h)")
    ax_occ1.set_ylabel("mask_area_t / union")
    ax_occ1.legend(loc="best", fontsize=7)
    ax_occ1.set_title("Mask occupancy vs time")
    ax_occ2.set_visible(False)
    out_path = _ensure_dir(out_dir) / "valid_coverage.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def _symmetric_vmin_vmax(m: np.ndarray, default: float = 1e-9) -> Tuple[float, float]:
    """Zero-centered symmetric color limits: m_abs = max(abs(finite)), return (-m_abs, +m_abs)."""
    finite = m[np.isfinite(m)]
    if len(finite) == 0:
        return -default, default
    m_abs = float(np.nanmax(np.abs(finite)))
    if m_abs <= 0:
        return -default, default
    return -m_abs, m_abs


def plot_persistent_norm_rate_maps(
    episodes: List[ParticleEpisode],
    summary: DescriptiveSummary,
    out_dir: str,
    use_global_scale: bool = False,
) -> str:
    """Plot time-mean persistent normalized rate (signed dc/dt / mean|dc/dt| per t) per particle.
    Color scale is symmetric about 0. Optional global scale across all panels."""
    from reaction_kinetics.config import PERSISTENT_NORM_RATE_USE_GLOBAL_SCALE
    use_global = use_global_scale or PERSISTENT_NORM_RATE_USE_GLOBAL_SCALE
    per_particle = summary.per_particle
    if not per_particle:
        return ""
    n = len(per_particle)
    ncol = min(4, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    if n == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    global_m_abs = 0.0
    if use_global:
        for p in per_particle:
            m = p.get("persistent_norm_rate_yx")
            if m is not None and np.any(np.isfinite(m)):
                v, _ = _symmetric_vmin_vmax(m)
                global_m_abs = max(global_m_abs, abs(v))
        if global_m_abs <= 0:
            global_m_abs = 1e-9
    for idx, p in enumerate(per_particle):
        ax = axes[idx]
        m = p.get("persistent_norm_rate_yx")
        if m is not None and np.any(np.isfinite(m)):
            if use_global:
                vmin, vmax = -global_m_abs, global_m_abs
            else:
                vmin, vmax = _symmetric_vmin_vmax(m)
            im = ax.imshow(m, aspect="equal", cmap="RdBu_r", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, label="Norm. rate")
        else:
            ax.set_visible(False)
        ax.set_title(f"Particle {p['particle_id']}")
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    out_path = _ensure_dir(out_dir) / "persistent_norm_rate_maps.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def plot_persistent_norm_rate_mean_vs_median(
    summary: DescriptiveSummary,
    out_dir: str,
) -> str:
    """Two rows per particle block: time-mean then time-median persistent normalized rate; symmetric scale."""
    per_particle = summary.per_particle
    if not per_particle:
        return ""
    n = len(per_particle)
    ncol = min(4, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(2 * nrow, ncol, figsize=(4 * ncol, 4 * 2 * nrow))
    if 2 * nrow == 1:
        axes = axes.reshape(1, -1)
    for idx, p in enumerate(per_particle):
        r_block = idx // ncol
        c = idx % ncol
        ax_mean = axes[2 * r_block, c]
        ax_median = axes[2 * r_block + 1, c]
        m_mean = p.get("persistent_norm_rate_yx")
        m_median = p.get("persistent_norm_rate_median_yx")
        for ax, m in [(ax_mean, m_mean), (ax_median, m_median)]:
            if m is not None and np.any(np.isfinite(m)):
                vmin, vmax = _symmetric_vmin_vmax(m)
                im = ax.imshow(m, aspect="equal", cmap="RdBu_r", vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, label="Norm. rate")
            ax.set_title(f"P{p['particle_id']}")
        ax_mean.set_ylabel("Mean")
        ax_median.set_ylabel("Median")
    for idx in range(n, nrow * ncol):
        c = idx % ncol
        r_block = idx // ncol
        axes[2 * r_block, c].set_visible(False)
        axes[2 * r_block + 1, c].set_visible(False)
    out_path = _ensure_dir(out_dir) / "persistent_norm_rate_mean_vs_median.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def plot_persistent_deviation_maps(
    summary: DescriptiveSummary,
    out_dir: str,
) -> str:
    """Per-particle map: time-mean of (norm_rate - spatial mean at each t). Symmetric scale about 0."""
    per_particle = summary.per_particle
    if not per_particle:
        return ""
    n = len(per_particle)
    ncol = min(4, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    if n == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    for idx, p in enumerate(per_particle):
        ax = axes[idx]
        m = p.get("persistent_deviation_norm_yx")
        if m is not None and np.any(np.isfinite(m)):
            vmin, vmax = _symmetric_vmin_vmax(m)
            im = ax.imshow(m, aspect="equal", cmap="RdBu_r", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, label="Deviation (norm.)")
        else:
            ax.set_visible(False)
        ax.set_title(f"Particle {p['particle_id']}\nDeviation from particle-mean")
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    out_path = _ensure_dir(out_dir) / "persistent_deviation_norm_maps.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def plot_boundary_vs_interior(
    summary: DescriptiveSummary,
    out_dir: str,
) -> str:
    """Mean SOC and mean |dc/dt| vs time for boundary vs interior."""
    fig, (ax_soc, ax_rate) = plt.subplots(2, 1, sharex=False, figsize=(10, 8))
    for p in summary.per_particle:
        pid = p["particle_id"]
        t_s = p.get("time_s")
        t_mid = p.get("time_mid_s")
        sb = p.get("mean_soc_boundary_t")
        si = p.get("mean_soc_interior_t")
        rb = p.get("mean_abs_rate_boundary_t")
        ri = p.get("mean_abs_rate_interior_t")
        if t_s is not None and sb is not None and si is not None and len(t_s) == len(sb) == len(si):
            ax_soc.plot(t_s / 3600, sb, alpha=0.7, ls="-", label=f"P{pid} boundary")
            ax_soc.plot(t_s / 3600, si, alpha=0.7, ls="--", label=f"P{pid} interior")
        if t_mid is not None and rb is not None and ri is not None and len(t_mid) == len(rb) == len(ri):
            ax_rate.plot(t_mid / 3600, rb, alpha=0.7, ls="-", label=f"P{pid} boundary")
            ax_rate.plot(t_mid / 3600, ri, alpha=0.7, ls="--", label=f"P{pid} interior")
    ax_soc.set_ylabel("Mean SOC")
    ax_soc.set_xlabel("Time (h)")
    ax_soc.set_title("Boundary vs interior: mean SOC vs time")
    ax_soc.legend(loc="best", fontsize=7)
    ax_rate.set_ylabel("Mean |dc/dt|")
    ax_rate.set_xlabel("Time (h)")
    ax_rate.set_title("Boundary vs interior: mean |dc/dt| vs time")
    ax_rate.legend(loc="best", fontsize=7)
    out_path = _ensure_dir(out_dir) / "boundary_vs_interior.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def _binned_trend(x: np.ndarray, y: np.ndarray, n_bins: int = 10, use_median: bool = True) -> tuple:
    """Return (bin_centers, trend_values) for bins on [0, 1]; trend is median or mean per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    dig = np.digitize(x, bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    trend = []
    for i in range(1, len(bins)):
        sel = dig == i
        if np.sum(sel) > 0:
            vals = y[sel]
            vals = vals[np.isfinite(vals)]
            trend.append(float(np.nanmedian(vals) if use_median else np.nanmean(vals)) if len(vals) else np.nan)
        else:
            trend.append(np.nan)
    return np.array(bin_centers), np.array(trend)


def plot_per_particle_dc_dt_vs_soc(
    episode: ParticleEpisode,
    rate_maps: RateMaps,
    out_dir: Path,
    particle_id: int,
) -> str:
    """Per-particle: 4 panels — dc/dt vs SOC and |dc/dt| vs SOC, each split rising/falling."""
    T = episode.soc_movie_tyx.shape[0]
    c_mid = (episode.soc_movie_tyx[:-1] + episode.soc_movie_tyx[1:]) / 2
    c_rise, dc_rise = [], []
    c_fall, dc_fall = [], []
    for t in range(T - 1):
        v = rate_maps.valid_mask_tyx[t] & np.isfinite(rate_maps.dc_dt_tyx[t]) & np.isfinite(c_mid[t])
        if not np.any(v):
            continue
        c_vals = c_mid[t][v].ravel()
        dc_vals = rate_maps.dc_dt_tyx[t][v].ravel()
        if episode.drive_sign[t] > 0:
            c_rise.extend(c_vals)
            dc_rise.extend(dc_vals)
        elif episode.drive_sign[t] < 0:
            c_fall.extend(c_vals)
            dc_fall.extend(dc_vals)
    c_rise, dc_rise = np.array(c_rise), np.array(dc_rise)
    c_fall, dc_fall = np.array(c_fall), np.array(dc_fall)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax_rise_dc, ax_fall_dc = axes[0, 0], axes[0, 1]
    ax_rise_abs, ax_fall_abs = axes[1, 0], axes[1, 1]
    for ax, all_c, all_dc, title, use_abs in [
        (ax_rise_dc, c_rise, dc_rise, "Rising: dc/dt vs SOC", False),
        (ax_fall_dc, c_fall, dc_fall, "Falling: dc/dt vs SOC", False),
        (ax_rise_abs, c_rise, np.abs(dc_rise), "Rising: |dc/dt| vs SOC", True),
        (ax_fall_abs, c_fall, np.abs(dc_fall), "Falling: |dc/dt| vs SOC", True),
    ]:
        if len(all_c) == 0:
            ax.set_title(title)
            ax.set_xlabel("SOC (midpoint)")
            ax.set_ylabel("|dc/dt|" if use_abs else "dc/dt")
            ax.axhline(0, color="gray", ls="--")
            continue
        if len(all_c) > 2000:
            idx = np.random.choice(len(all_c), 2000, replace=False)
            ax.scatter(all_c[idx], all_dc[idx], alpha=0.15, s=5)
        else:
            ax.scatter(all_c, all_dc, alpha=0.25, s=5)
        if len(all_c) > 10:
            bc, tr = _binned_trend(all_c, all_dc, use_median=True)
            if np.any(np.isfinite(tr)):
                ax.plot(bc, tr, "k-", lw=2, label="Binned median")
        ax.set_xlabel("SOC (midpoint)")
        ax.set_ylabel("|dc/dt|" if use_abs else "dc/dt")
        ax.set_title(title)
        ax.axhline(0, color="gray", ls="--")
    out_path = out_dir / "dc_dt_vs_soc.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def plot_per_particle_mean_summaries(
    per_particle_dict: Dict[str, Any],
    out_dir: Path,
    particle_id: int,
) -> str:
    """Per-particle: mean SOC vs t, mean dc/dt vs t, mean |dc/dt| vs t, mean |dc/dt| vs mean SOC."""
    time_s = per_particle_dict.get("time_s")
    time_mid_s = per_particle_dict.get("time_mid_s")
    mean_soc_t = per_particle_dict.get("mean_soc_t")
    mean_dc_dt_t = per_particle_dict.get("mean_dc_dt_t")
    mean_abs_rate_t = per_particle_dict.get("mean_abs_rate_t")
    if time_s is None or mean_soc_t is None:
        return ""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    ax1.plot(time_s / 3600, mean_soc_t, "b-")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Mean SOC")
    ax1.set_title("Mean SOC vs time")
    if time_mid_s is not None and mean_dc_dt_t is not None:
        ax2.plot(time_mid_s / 3600, mean_dc_dt_t, "b-")
        ax2.axhline(0, color="gray", ls="--")
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Mean dc/dt")
    ax2.set_title("Mean signed dc/dt vs time")
    if time_mid_s is not None and mean_abs_rate_t is not None:
        ax3.plot(time_mid_s / 3600, mean_abs_rate_t, "b-")
    ax3.set_xlabel("Time (h)")
    ax3.set_ylabel("Mean |dc/dt|")
    ax3.set_title("Mean |dc/dt| vs time")
    if mean_soc_t is not None and mean_abs_rate_t is not None and len(mean_soc_t) >= 2 and len(mean_abs_rate_t) >= 1:
        soc_mid = (mean_soc_t[:-1] + mean_soc_t[1:]) / 2
        ax4.plot(soc_mid, mean_abs_rate_t, "b-")
    ax4.set_xlabel("Mean SOC (midpoint)")
    ax4.set_ylabel("Mean |dc/dt|")
    ax4.set_title("Mean |dc/dt| vs mean SOC")
    out_path = out_dir / "particle_mean_summaries.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def plot_per_particle_boundary_vs_interior(
    per_particle_dict: Dict[str, Any],
    out_dir: Path,
    particle_id: int,
) -> str:
    """Per-particle: boundary vs interior SOC and |dc/dt| vs time."""
    t_s = per_particle_dict.get("time_s")
    t_mid = per_particle_dict.get("time_mid_s")
    sb = per_particle_dict.get("mean_soc_boundary_t")
    si = per_particle_dict.get("mean_soc_interior_t")
    rb = per_particle_dict.get("mean_abs_rate_boundary_t")
    ri = per_particle_dict.get("mean_abs_rate_interior_t")
    fig, (ax_soc, ax_rate) = plt.subplots(2, 1, figsize=(8, 6))
    if t_s is not None and sb is not None and si is not None:
        ax_soc.plot(t_s / 3600, sb, "b-", label="Boundary")
        ax_soc.plot(t_s / 3600, si, "orange", ls="--", label="Interior")
    ax_soc.set_ylabel("Mean SOC")
    ax_soc.set_xlabel("Time (h)")
    ax_soc.set_title("Boundary vs interior: mean SOC vs time")
    ax_soc.legend()
    if t_mid is not None and rb is not None and ri is not None:
        ax_rate.plot(t_mid / 3600, rb, "b-", label="Boundary")
        ax_rate.plot(t_mid / 3600, ri, "orange", ls="--", label="Interior")
    ax_rate.set_ylabel("Mean |dc/dt|")
    ax_rate.set_xlabel("Time (h)")
    ax_rate.set_title("Boundary vs interior: mean |dc/dt| vs time")
    ax_rate.legend()
    out_path = out_dir / "boundary_vs_interior.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def plot_per_particle_persistent_norm_rate(
    per_particle_dict: Dict[str, Any],
    out_dir: Path,
    particle_id: int,
) -> str:
    """Per-particle: time-mean persistent normalized rate; symmetric color scale about 0."""
    m = per_particle_dict.get("persistent_norm_rate_yx")
    if m is None or not np.any(np.isfinite(m)):
        return ""
    fig, ax = plt.subplots()
    vmin, vmax = _symmetric_vmin_vmax(m)
    im = ax.imshow(m, aspect="equal", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Norm. rate")
    ax.set_title(f"Particle {particle_id}: persistent normalized rate")
    out_path = out_dir / "persistent_norm_rate.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def plot_per_particle_persistent_norm_rate_segments(
    per_particle_dict: Dict[str, Any],
    out_dir: Path,
    particle_id: int,
) -> str:
    """Per-particle: 4 panels for early_rising, late_rising, early_falling, late_falling segment norm rate maps."""
    seg_maps = per_particle_dict.get("segment_mean_norm_rate_yx") or {}
    if not seg_maps:
        return ""
    seg_names = ["early_rising", "late_rising", "early_falling", "late_falling"]
    maps_to_plot = []
    for name in seg_names:
        arr = seg_maps.get(name)
        if arr is not None and np.any(np.isfinite(arr)):
            maps_to_plot.append((name, arr))
    if not maps_to_plot:
        return ""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    vmin, vmax = None, None
    for _, arr in maps_to_plot:
        finite = arr[np.isfinite(arr)]
        if len(finite):
            p2, p98 = np.nanpercentile(finite, [2, 98])
            vmin = p2 if vmin is None else min(vmin, p2)
            vmax = p98 if vmax is None else max(vmax, p98)
    # Symmetric about 0 for diverging colormap
    if vmin is not None and vmax is not None:
        m_abs = max(abs(vmin), abs(vmax))
        if m_abs > 0:
            vmin, vmax = -m_abs, m_abs
    for idx, (name, arr) in enumerate(maps_to_plot):
        ax = axes[idx]
        im = ax.imshow(arr, aspect="equal", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        plt.colorbar(im, ax=ax, label="Norm. rate")
    for j in range(len(maps_to_plot), 4):
        axes[j].set_visible(False)
    out_path = out_dir / "persistent_norm_rate_segments.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def plot_per_particle_diagnostics(
    episodes: List[ParticleEpisode],
    rate_maps_list: List[RateMaps],
    summary: DescriptiveSummary,
    out_dir: str,
) -> List[str]:
    """Generate all per-particle figures and save norm_rate_tyx.npy per particle. Returns list of paths."""
    paths: List[str] = []
    base = _ensure_dir(out_dir) / "per_particle"
    for i, (ep, rate, p) in enumerate(zip(episodes, rate_maps_list, summary.per_particle)):
        pid = p["particle_id"]
        particle_dir = base / f"particle_{pid}"
        particle_dir.mkdir(parents=True, exist_ok=True)
        paths.append(plot_per_particle_dc_dt_vs_soc(ep, rate, particle_dir, pid))
        p1 = plot_per_particle_mean_summaries(p, particle_dir, pid)
        if p1:
            paths.append(p1)
        paths.append(plot_per_particle_boundary_vs_interior(p, particle_dir, pid))
        p2 = plot_per_particle_persistent_norm_rate(p, particle_dir, pid)
        if p2:
            paths.append(p2)
        p3 = plot_per_particle_persistent_norm_rate_segments(p, particle_dir, pid)
        if p3:
            paths.append(p3)
        norm_tyx = p.get("norm_rate_tyx")
        if norm_tyx is not None and isinstance(norm_tyx, np.ndarray):
            npy_path = particle_dir / "norm_rate_tyx.npy"
            np.save(npy_path, norm_tyx)
            paths.append(str(npy_path))
    return paths


def plot_all(
    episodes: List[ParticleEpisode],
    rate_maps_list: List[RateMaps],
    summary: DescriptiveSummary,
    out_dir: str,
) -> List[str]:
    """Generate all figures. Returns list of saved paths."""
    paths = []
    paths.append(plot_particle_soc_curves(episodes, summary, out_dir))
    paths.append(plot_dc_dt_vs_c(episodes, rate_maps_list, out_dir))
    paths.append(plot_heterogeneity_vs_time(summary, out_dir))
    paths.append(plot_valid_coverage(summary, out_dir))
    p1 = plot_persistent_norm_rate_maps(episodes, summary, out_dir)
    if p1:
        paths.append(p1)
    p2 = plot_persistent_norm_rate_mean_vs_median(summary, out_dir)
    if p2:
        paths.append(p2)
    p3 = plot_persistent_deviation_maps(summary, out_dir)
    if p3:
        paths.append(p3)
    paths.append(plot_boundary_vs_interior(summary, out_dir))
    paths.extend(plot_per_particle_diagnostics(episodes, rate_maps_list, summary, out_dir))
    return paths


# ----- Boundary-only kinetics (particles 2, 4, 5, 6, 7) -----


def plot_boundary_only_dc_dt_vs_soc_per_particle(
    per_particle: Dict[int, Dict[str, Any]],
    out_dir: Path,
) -> List[str]:
    """Per-particle boundary-only: 4 panels (rising/falling x signed/abs) with scatter + binned median."""
    from reaction_kinetics.boundary_kinetics import binned_fit_on_grid
    paths: List[str] = []
    base = out_dir / "boundary_analysis"
    soc_grid = np.linspace(0, 1, 11)
    for pid, data in per_particle.items():
        particle_dir = base / f"particle_{pid}"
        particle_dir.mkdir(parents=True, exist_ok=True)
        c_rise, dc_rise = data["rising"]
        c_fall, dc_fall = data["falling"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for ax, all_c, all_dc, title, use_abs in [
            (axes[0, 0], c_rise, dc_rise, "Rising: dc/dt vs SOC (boundary)", False),
            (axes[0, 1], c_fall, dc_fall, "Falling: dc/dt vs SOC (boundary)", False),
            (axes[1, 0], c_rise, np.abs(dc_rise), "Rising: |dc/dt| vs SOC (boundary)", True),
            (axes[1, 1], c_fall, np.abs(dc_fall), "Falling: |dc/dt| vs SOC (boundary)", True),
        ]:
            if len(all_c) == 0:
                ax.set_title(title)
                ax.set_xlabel("SOC (midpoint)")
                ax.set_ylabel("|dc/dt|" if use_abs else "dc/dt")
                ax.axhline(0, color="gray", ls="--")
                continue
            if len(all_c) > 1500:
                idx = np.random.choice(len(all_c), 1500, replace=False)
                ax.scatter(all_c[idx], all_dc[idx], alpha=0.2, s=5)
            else:
                ax.scatter(all_c, all_dc, alpha=0.3, s=5)
            if len(all_c) > 5:
                fit = binned_fit_on_grid(all_c, all_dc, soc_grid, use_median=True)
                if np.any(np.isfinite(fit)):
                    ax.plot(soc_grid, fit, "k-", lw=2, label="Binned median")
            ax.set_xlabel("SOC (midpoint)")
            ax.set_ylabel("|dc/dt|" if use_abs else "dc/dt")
            ax.set_title(title)
            ax.axhline(0, color="gray", ls="--")
        out_path = particle_dir / "boundary_dc_dt_vs_soc.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(str(out_path))
    return paths


def plot_boundary_only_pooled(
    pooled_normalized: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
) -> List[str]:
    """Pooled boundary-only (rate-normalized per particle): 4 panels signed + abs, rising + falling."""
    from reaction_kinetics.boundary_kinetics import binned_fit_on_grid
    paths: List[str] = []
    base = _ensure_dir(str(out_dir)) / "boundary_analysis"
    soc_grid = np.linspace(0, 1, 51)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, seg, rate_type, use_abs in [
        (axes[0, 0], "rising", "signed", False),
        (axes[0, 1], "falling", "signed", False),
        (axes[1, 0], "rising", "abs", True),
        (axes[1, 1], "falling", "abs", True),
    ]:
        c, dc = pooled_normalized[seg]
        y = np.abs(dc) if use_abs else dc
        if len(c) == 0:
            ax.set_title(f"Boundary pooled ({seg}, {rate_type})")
            ax.set_xlabel("SOC (midpoint)")
            ax.axhline(0, color="gray", ls="--")
            continue
        if len(c) > 3000:
            idx = np.random.choice(len(c), 3000, replace=False)
            ax.scatter(c[idx], y[idx], alpha=0.1, s=5)
        else:
            ax.scatter(c, y, alpha=0.15, s=5)
        if len(c) > 10:
            fit = binned_fit_on_grid(c, y, soc_grid, use_median=True)
            if np.any(np.isfinite(fit)):
                ax.plot(soc_grid, fit, "k-", lw=2, label="Binned median")
        ax.set_xlabel("SOC (midpoint)")
        ax.set_ylabel("|dc/dt| (norm)" if use_abs else "dc/dt (norm)")
        ax.set_title(f"Boundary only pooled ({seg}, {rate_type})")
        ax.axhline(0, color="gray", ls="--")
    out_path = base / "boundary_pooled.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(str(out_path))
    return paths


def plot_boundary_vs_full_fits(
    soc_grid: np.ndarray,
    boundary_fits: Dict[str, Dict[str, np.ndarray]],
    full_fits: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
) -> str:
    """Comparison: boundary-only fit vs full-particle fit on same axes (4 panels)."""
    base = _ensure_dir(str(out_dir)) / "boundary_analysis"
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, seg, rate_type in [
        (axes[0, 0], "rising", "signed"),
        (axes[0, 1], "falling", "signed"),
        (axes[1, 0], "rising", "abs"),
        (axes[1, 1], "falling", "abs"),
    ]:
        bf = boundary_fits[seg].get(rate_type)
        ff = full_fits[seg].get(rate_type)
        if bf is not None and np.any(np.isfinite(bf)):
            ax.plot(soc_grid, bf, "b-", lw=2, label="Boundary fit")
        if ff is not None and np.any(np.isfinite(ff)):
            ax.plot(soc_grid, ff, "orange", ls="--", lw=2, label="Full-particle fit")
        ax.set_xlabel("SOC")
        ax.set_ylabel("dc/dt" if rate_type == "signed" else "|dc/dt|")
        ax.set_title(f"{seg}: boundary vs full-particle fit")
        ax.legend()
        ax.axhline(0, color="gray", ls=":")
    out_path = base / "boundary_vs_full_particle_fits.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)
