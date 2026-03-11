"""
Diagnostic plots. dc/dt has length T-1 and is aligned to time_mid_s (midpoints).
"""

from pathlib import Path
from typing import List

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
    return paths
