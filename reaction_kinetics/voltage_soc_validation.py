"""
Voltage–SOC proxy validation: compare electrochemical voltage to Raman-derived SOC over time.
Goal: show shared temporal structure, reasonable lag, aligned turning points, interpretable V–SOC relationship.
Uses retained particles only (e.g. exclude 1, 3, 8). All-particle mean SOC is the primary validation signal.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import argrelmax, argrelmin
except ImportError:
    argrelmax = argrelmin = None

from reaction_kinetics.schema import DescriptiveSummary, ParticleEpisode

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _normalize01(x: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] over finite values."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.full_like(x, np.nan)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi <= lo:
        return np.where(finite, 0.5, np.nan)
    out = np.full_like(x, np.nan)
    out[finite] = (x[finite] - lo) / (hi - lo)
    return out


def _filter_retained(
    episodes: List[ParticleEpisode],
    summary: DescriptiveSummary,
    exclude_particle_ids: List[int],
) -> Tuple[List[ParticleEpisode], List[Dict], np.ndarray, np.ndarray, np.ndarray]:
    """Return (retained_episodes, retained_per_particle, time_s, voltage_v, retained_mean_soc_t)."""
    retained_ep, retained_pp = [], []
    for ep, p in zip(episodes, summary.per_particle):
        if ep.particle_id in exclude_particle_ids:
            continue
        retained_ep.append(ep)
        retained_pp.append(p)
    if not retained_ep:
        return retained_ep, retained_pp, np.array([]), np.array([]), np.array([])
    time_s = retained_ep[0].time_s
    voltage_v = retained_ep[0].voltage_v
    stacked = np.array([p["mean_soc_t"] for p in retained_pp])
    retained_mean_soc_t = np.nanmean(stacked, axis=0)
    return retained_ep, retained_pp, time_s, voltage_v, retained_mean_soc_t


def _segment_bounds(drive_sign: np.ndarray) -> List[Tuple[int, int, int]]:
    """List of (start_idx, end_idx, sign) for contiguous blocks of +1 or -1."""
    T = len(drive_sign)
    out = []
    i = 0
    while i < T:
        s = drive_sign[i]
        if s == 0:
            i += 1
            continue
        start = i
        while i < T and drive_sign[i] == s:
            i += 1
        out.append((start, i, int(s)))
    return out


def _crosscorr_lag(
    soc: np.ndarray,
    voltage: np.ndarray,
    time_s: np.ndarray,
) -> Tuple[float, int, float]:
    """Cross-correlation of soc vs voltage. Returns (correlation_peak, best_lag_timestep, best_lag_hours).
    Convention: positive lag = SOC lags voltage (SOC peak later than voltage peak)."""
    soc_n = _normalize01(soc)
    v_n = _normalize01(voltage)
    valid = np.isfinite(soc_n) & np.isfinite(v_n)
    if np.sum(valid) < 3:
        return np.nan, 0, np.nan
    soc_n = np.nan_to_num(soc_n, nan=0.0)
    v_n = np.nan_to_num(v_n, nan=0.0)
    corr = np.correlate(soc_n, v_n, mode="full")
    T = len(soc_n)
    lag_timestep = int(np.argmax(corr) - (T - 1))
    peak = float(corr[T - 1 + lag_timestep]) / (np.sqrt(np.sum(soc_n**2) * np.sum(v_n**2)) + 1e-12)
    dt_s = np.median(np.diff(time_s)) if len(time_s) > 1 else 0.0
    lag_hours = lag_timestep * dt_s / 3600.0 if np.isfinite(dt_s) else np.nan
    return peak, lag_timestep, lag_hours


def _fit_linear_r2(t: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Linear fit y = a*t + b. Returns (slope, intercept, r2)."""
    valid = np.isfinite(t) & np.isfinite(y)
    if np.sum(valid) < 2:
        return np.nan, np.nan, np.nan
    t_, y_ = t[valid], y[valid]
    n = len(t_)
    A = np.column_stack([t_, np.ones(n)])
    coeffs, *_ = np.linalg.lstsq(A, y_, rcond=None)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    y_pred = slope * t_ + intercept
    ss_tot = np.sum((y_ - np.mean(y_)) ** 2)
    ss_res = np.sum((y_ - y_pred) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return slope, intercept, r2


def _segment_fits(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    soc_t: np.ndarray,
    drive_sign: np.ndarray,
) -> List[Dict[str, Any]]:
    """Linear fits per monotonic segment. Returns list of dicts with segment_index, segment_type, t_start, t_end, voltage_slope, soc_slope, r2_voltage, r2_soc."""
    segments = _segment_bounds(drive_sign)
    rows = []
    for idx, (start, end, sign) in enumerate(segments):
        if end - start < 2:
            continue
        t_seg = time_s[start:end]
        v_seg = voltage_v[start:end]
        s_seg = soc_t[start:end]
        vs, vi, r2v = _fit_linear_r2(t_seg, v_seg)
        ss, si, r2s = _fit_linear_r2(t_seg, s_seg)
        rows.append({
            "segment_index": idx,
            "segment_type": "rising" if sign == 1 else "falling",
            "t_start": start,
            "t_end": end,
            "voltage_slope": vs,
            "soc_slope": ss,
            "r2_voltage": r2v,
            "r2_soc": r2s,
        })
    return rows


def _turning_points(y: np.ndarray, time_s: np.ndarray, order: int = 2) -> List[Tuple[int, float, str]]:
    """Local maxima and minima. Returns list of (timestep, time_h, "max"|"min")."""
    out = []
    if argrelmax is not None and argrelmin is not None and len(y) > order * 2:
        for idx in argrelmax(y, order=order)[0]:
            if np.isfinite(y[idx]):
                out.append((int(idx), float(time_s[idx]) / 3600.0, "max"))
        for idx in argrelmin(y, order=order)[0]:
            if np.isfinite(y[idx]):
                out.append((int(idx), float(time_s[idx]) / 3600.0, "min"))
    out.sort(key=lambda x: x[0])
    return out


def _align_turning_points(
    volt_events: List[Tuple[int, float, str]],
    soc_events: List[Tuple[int, float, str]],
) -> List[Dict[str, Any]]:
    """For each voltage event, find nearest SOC event of same type; compute offset (soc_time - voltage_time). Positive = SOC later."""
    rows = []
    for (vt, vh, etype) in volt_events:
        same = [(st, sh) for st, sh, et in soc_events if et == etype]
        if not same:
            rows.append({"event_type": etype, "voltage_timestep": vt, "voltage_time_h": vh, "soc_timestep": None, "soc_time_h": np.nan, "offset_hours": np.nan})
            continue
        nearest = min(same, key=lambda x: abs(x[1] - vh))
        st, sh = nearest
        rows.append({"event_type": etype, "voltage_timestep": vt, "voltage_time_h": vh, "soc_timestep": st, "soc_time_h": sh, "offset_hours": sh - vh})
    return rows


def _fft_metrics(time_s: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Dominant period (hours) and spectral centroid (1/hours). Uses magnitude spectrum."""
    valid = np.isfinite(y)
    if np.sum(valid) < 4:
        return np.nan, np.nan
    y_ = np.nan_to_num(y, nan=np.nanmean(y))
    dt = float(np.median(np.diff(time_s)))
    if dt <= 0:
        return np.nan, np.nan
    n = len(y_)
    spec = np.abs(np.fft.rfft(y_ - np.mean(y_)))
    freqs = np.fft.rfftfreq(n, d=dt)
    if len(freqs) < 2:
        return np.nan, np.nan
    skip = 1
    dom_idx = skip + np.argmax(spec[skip:])
    dom_freq = float(freqs[dom_idx])
    period_h = (1.0 / dom_freq) / 3600.0 if dom_freq > 0 else np.nan
    power = spec ** 2
    centroid = np.sum(freqs * power) / (np.sum(power) + 1e-12)
    centroid_h = centroid * 3600.0
    return period_h, centroid_h


def _compute_inverted_voltage(voltage_v: np.ndarray) -> np.ndarray:
    """Normalize voltage to [0,1] over the run and return inverted normalized voltage.

    For an NMC cathode, higher voltage corresponds to lower cathode lithiation (lower SOC),
    so the Raman SOC proxy should align with 1 - normalized voltage.
    """
    v_norm = _normalize01(voltage_v)
    return 1.0 - v_norm


def _segmentwise_crosscorr(
    soc: np.ndarray,
    v_inv_norm: np.ndarray,
    time_s: np.ndarray,
    segments: List[Tuple[int, int, int]],
    max_lag_fraction: float = 0.25,
) -> List[Dict[str, Any]]:
    """Segment-wise cross-correlation between SOC and inverted voltage.

    Returns one dict per segment with:
    segment_index, segment_type, segment_start_timestep, segment_end_timestep,
    corr_peak, best_lag_timestep, best_lag_hours.

    Convention: positive lag = SOC lags inverted voltage (SOC peak later in time).
    """
    rows: List[Dict[str, Any]] = []
    if len(soc) != len(v_inv_norm) or len(time_s) != len(soc):
        return rows

    dt_s = np.median(np.diff(time_s)) if len(time_s) > 1 else 0.0
    for seg_id, (start, end, sign) in enumerate(segments):
        length = end - start
        if length < 4:
            continue
        s_seg = soc[start:end]
        v_seg = v_inv_norm[start:end]
        valid = np.isfinite(s_seg) & np.isfinite(v_seg)
        if np.sum(valid) < 4:
            continue
        s_seg = s_seg[valid]
        v_seg = v_seg[valid]
        if len(s_seg) < 4 or len(v_seg) < 4:
            continue

        # Normalize within segment for stable correlation
        s_seg = (s_seg - np.mean(s_seg)) / (np.std(s_seg) + 1e-12)
        v_seg = (v_seg - np.mean(v_seg)) / (np.std(v_seg) + 1e-12)

        n = len(s_seg)
        full_corr = np.correlate(s_seg, v_seg, mode="full")
        # lags from -(n-1) to +(n-1)
        all_lags = np.arange(-(n - 1), n)
        max_lag = max(1, int(max_lag_fraction * n))
        mask = (all_lags >= -max_lag) & (all_lags <= max_lag)
        if not np.any(mask):
            continue
        corr_window = full_corr[mask]
        lags_window = all_lags[mask]
        best_idx = int(np.argmax(corr_window))
        best_lag = int(lags_window[best_idx])
        peak = float(corr_window[best_idx]) / (np.sqrt(np.sum(s_seg**2) * np.sum(v_seg**2)) + 1e-12)
        lag_hours = best_lag * dt_s / 3600.0 if dt_s > 0 else np.nan

        rows.append(
            {
                "segment_id": seg_id,
                "segment_type": "rising" if sign == 1 else "falling",
                "segment_start_timestep": int(start),
                "segment_end_timestep": int(end),
                "corr_peak": peak,
                "best_lag_timestep": best_lag,
                "best_lag_hours": lag_hours,
            }
        )
    return rows


def _inversion_aware_turning_points(
    voltage_v: np.ndarray,
    soc: np.ndarray,
    time_s: np.ndarray,
) -> List[Dict[str, Any]]:
    """Pair voltage maxima with SOC minima and voltage minima with SOC maxima.

    Returns per-event rows with particle-agnostic alignment; caller attaches particle_id.
    """
    volt_events = _turning_points(voltage_v, time_s)
    soc_events = _turning_points(soc, time_s)
    rows: List[Dict[str, Any]] = []
    if not volt_events or not soc_events:
        return rows

    # Build separate lists for SOC max/min
    soc_max = [(idx, t_h) for idx, t_h, et in soc_events if et == "max"]
    soc_min = [(idx, t_h) for idx, t_h, et in soc_events if et == "min"]

    for v_idx, v_time_h, v_type in volt_events:
        if v_type == "max":
            pool = soc_min  # inverted: voltage max ↔ SOC min
            matched_type = "min"
        else:
            pool = soc_max  # voltage min ↔ SOC max
            matched_type = "max"
        if not pool:
            rows.append(
                {
                    "voltage_event_type": v_type,
                    "matched_soc_event_type": matched_type,
                    "voltage_timestep": v_idx,
                    "soc_timestep": np.nan,
                    "offset_timestep": np.nan,
                    "offset_hours": np.nan,
                }
            )
            continue
        nearest = min(pool, key=lambda x: abs(x[1] - v_time_h))
        s_idx, s_time_h = nearest
        offset_hours = s_time_h - v_time_h
        rows.append(
            {
                "voltage_event_type": v_type,
                "matched_soc_event_type": matched_type,
                "voltage_timestep": int(v_idx),
                "soc_timestep": int(s_idx),
                "offset_timestep": int(s_idx - v_idx),
                "offset_hours": offset_hours,
            }
        )
    return rows


def _plot_overlays(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    retained_ep: List[ParticleEpisode],
    retained_pp: List[Dict],
    retained_mean_soc_t: np.ndarray,
    out_dir: Path,
) -> List[str]:
    paths = []
    time_h = time_s / 3600.0
    v_n = _normalize01(voltage_v)
    for ep, p in zip(retained_ep, retained_pp):
        soc = p["mean_soc_t"]
        s_n = _normalize01(soc)
        fig, ax = plt.subplots()
        ax.plot(time_h, s_n, label="SOC (norm.)", alpha=0.8)
        ax.plot(time_h, v_n, label="Voltage (norm.)", alpha=0.8)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized [0,1]")
        ax.legend()
        ax.set_title(f"Particle {ep.particle_id}: Voltage vs SOC overlay")
        out_path = out_dir / f"overlay_particle_{ep.particle_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(str(out_path))
    if len(retained_mean_soc_t):
        s_global = _normalize01(retained_mean_soc_t)
        fig, ax = plt.subplots()
        ax.plot(time_h, s_global, label="SOC (norm.)", alpha=0.8)
        ax.plot(time_h, v_n, label="Voltage (norm.)", alpha=0.8)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized [0,1]")
        ax.legend()
        ax.set_title("Global (retained mean): Voltage vs SOC overlay")
        out_path = out_dir / "overlay_global.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(str(out_path))
    return paths


def _plot_inverted_overlays(
    time_s: np.ndarray,
    v_inv_norm: np.ndarray,
    retained_ep: List[ParticleEpisode],
    retained_pp: List[Dict],
    retained_mean_soc_t: np.ndarray,
    out_dir: Path,
) -> List[str]:
    """Overlay inverted normalized voltage with normalized SOC (global and per-particle)."""
    paths: List[str] = []
    time_h = time_s / 3600.0

    # Global
    if len(retained_mean_soc_t):
        s_global = _normalize01(retained_mean_soc_t)
        fig, ax = plt.subplots()
        ax.plot(time_h, s_global, label="SOC (norm.)", alpha=0.8)
        ax.plot(time_h, v_inv_norm, label="Inverted voltage (norm.)", alpha=0.8)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized [0,1]")
        ax.legend()
        ax.set_title("Global: SOC vs inverted voltage overlay")
        out_path = out_dir / "overlay_global_inverted_voltage.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(str(out_path))

    # Per-particle sheet (grid)
    n = len(retained_ep)
    if n:
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[n:]:
            ax.axis("off")
        for ax, ep, p in zip(axes, retained_ep, retained_pp):
            soc = p["mean_soc_t"]
            s_n = _normalize01(soc)
            ax.plot(time_h, s_n, label="SOC (norm.)", alpha=0.8)
            ax.plot(time_h, v_inv_norm, label="Inverted V (norm.)", alpha=0.8)
            ax.set_title(f"Particle {ep.particle_id}")
        for ax in axes[:n]:
            ax.set_xlabel("Time (h)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle("Per-particle SOC vs inverted voltage overlays")
        out_path = out_dir / "overlay_per_particle_inverted_voltage.png"
        fig.tight_layout(rect=[0, 0, 0.85, 0.95])
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(str(out_path))

    return paths


def _plot_crosscorr_global(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    retained_mean_soc_t: np.ndarray,
    out_dir: Path,
) -> str:
    soc_n = _normalize01(retained_mean_soc_t)
    v_n = _normalize01(voltage_v)
    soc_n = np.nan_to_num(soc_n, nan=0.0)
    v_n = np.nan_to_num(v_n, nan=0.0)
    corr = np.correlate(soc_n, v_n, mode="full")
    T = len(soc_n)
    lags = np.arange(-(T - 1), T)
    dt_s = np.median(np.diff(time_s)) if len(time_s) > 1 else 0.0
    lag_h = lags * dt_s / 3600.0
    fig, ax = plt.subplots()
    ax.plot(lag_h, corr)
    ax.axvline(0, color="gray", ls="--")
    ax.set_xlabel("Lag (hours). Positive = SOC lags voltage")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("Global: SOC vs voltage cross-correlation")
    out_path = out_dir / "crosscorr_global.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def _plot_voltage_vs_soc(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    retained_ep: List[ParticleEpisode],
    retained_pp: List[Dict],
    retained_mean_soc_t: np.ndarray,
    drive_sign: np.ndarray,
    out_dir: Path,
) -> List[str]:
    paths = []
    time_h = time_s / 3600.0
    for ep, p in zip(retained_ep, retained_pp):
        soc = p["mean_soc_t"]
        fig, ax = plt.subplots()
        scatter = ax.scatter(soc, voltage_v, c=time_h, cmap="viridis", alpha=0.6, s=10)
        ax.set_xlabel("Particle-mean SOC")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(f"Particle {ep.particle_id}: Voltage vs SOC")
        plt.colorbar(scatter, ax=ax, label="Time (h)")
        out_path = out_dir / f"voltage_vs_soc_particle_{ep.particle_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(str(out_path))
    if len(retained_mean_soc_t):
        fig, ax = plt.subplots()
        sc = ax.scatter(retained_mean_soc_t, voltage_v, c=time_h, cmap="viridis", alpha=0.6, s=10)
        ax.set_xlabel("All-particle mean SOC (retained)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("Global: Voltage vs SOC")
        plt.colorbar(sc, ax=ax, label="Time (h)")
        out_path = out_dir / "voltage_vs_soc_global.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(str(out_path))
    return paths


def _plot_spectrum(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    retained_mean_soc_t: np.ndarray,
    out_dir: Path,
) -> str:
    dt = float(np.median(np.diff(time_s)))
    if dt <= 0 or len(time_s) < 4:
        return ""
    v_ = voltage_v - np.nanmean(voltage_v)
    v_ = np.nan_to_num(v_, nan=0.0)
    s_ = retained_mean_soc_t - np.nanmean(retained_mean_soc_t)
    s_ = np.nan_to_num(s_, nan=0.0)
    n = len(time_s)
    spec_v = np.abs(np.fft.rfft(v_))
    spec_s = np.abs(np.fft.rfft(s_))
    freqs = np.fft.rfftfreq(n, d=dt)
    freq_h = freqs * 3600.0
    fig, ax = plt.subplots()
    ax.plot(freq_h[1:], spec_v[1:], label="Voltage", alpha=0.8)
    ax.plot(freq_h[1:], spec_s[1:], label="SOC", alpha=0.8)
    ax.set_xlabel("Frequency (1/h)")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.set_title("Spectrum comparison (global)")
    out_path = out_dir / "spectrum_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def _plot_spectrum_inverted(
    time_s: np.ndarray,
    v_inv_norm: np.ndarray,
    retained_mean_soc_t: np.ndarray,
    out_dir: Path,
) -> str:
    """Spectrum comparison for inverted voltage and SOC."""
    dt = float(np.median(np.diff(time_s)))
    if dt <= 0 or len(time_s) < 4:
        return ""
    v_ = v_inv_norm - np.nanmean(v_inv_norm)
    v_ = np.nan_to_num(v_, nan=0.0)
    s_ = retained_mean_soc_t - np.nanmean(retained_mean_soc_t)
    s_ = np.nan_to_num(s_, nan=0.0)
    n = len(time_s)
    spec_v = np.abs(np.fft.rfft(v_))
    spec_s = np.abs(np.fft.rfft(s_))
    freqs = np.fft.rfftfreq(n, d=dt)
    freq_h = freqs * 3600.0
    fig, ax = plt.subplots()
    ax.plot(freq_h[1:], spec_v[1:], label="Inverted voltage", alpha=0.8)
    ax.plot(freq_h[1:], spec_s[1:], label="SOC", alpha=0.8)
    ax.set_xlabel("Frequency (1/h)")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.set_title("Spectrum comparison (inverted voltage vs SOC)")
    out_path = out_dir / "spectrum_comparison_inverted_voltage.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def _plot_inverted_voltage_vs_soc_global(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    retained_mean_soc_t: np.ndarray,
    out_dir: Path,
) -> Optional[str]:
    """Global inverted normalized voltage vs SOC curve."""
    if not len(retained_mean_soc_t):
        return None
    time_h = time_s / 3600.0
    v_inv_norm = _compute_inverted_voltage(voltage_v)
    soc_n = _normalize01(retained_mean_soc_t)
    fig, ax = plt.subplots()
    sc = ax.scatter(soc_n, v_inv_norm, c=time_h, cmap="viridis", s=10, alpha=0.7)
    ax.set_xlabel("All-particle mean SOC (norm.)")
    ax.set_ylabel("Inverted voltage (norm.)")
    ax.set_title("Global: inverted normalized voltage vs SOC")
    plt.colorbar(sc, ax=ax, label="Time (h)")
    out_path = out_dir / "inverted_voltage_vs_soc_global.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def run_voltage_soc_validation(
    episodes: List[ParticleEpisode],
    summary: DescriptiveSummary,
    out_figures: Path,
    out_tables: Path,
    exclude_particle_ids: List[int],
) -> Tuple[List[str], List[str]]:
    """Run voltage–SOC proxy validation, including inversion-aware analyses.

    Returns (figure_paths, table_paths).
    """
    retained_ep, retained_pp, time_s, voltage_v, retained_mean_soc_t = _filter_retained(
        episodes, summary, exclude_particle_ids
    )
    if not retained_ep:
        return [], []

    out_dir = Path(out_figures) / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = Path(out_tables)
    figure_paths: List[str] = []
    drive_sign = retained_ep[0].drive_sign
    v_inv_norm = _compute_inverted_voltage(voltage_v)

    # 1. Overlays (raw voltage) and inversion-aware overlays
    figure_paths.extend(_plot_overlays(time_s, voltage_v, retained_ep, retained_pp, retained_mean_soc_t, out_dir))
    figure_paths.extend(_plot_inverted_overlays(time_s, v_inv_norm, retained_ep, retained_pp, retained_mean_soc_t, out_dir))

    # 2. Cross-correlation
    corr_rows = []
    for ep, p in zip(retained_ep, retained_pp):
        peak, lag_t, lag_h = _crosscorr_lag(p["mean_soc_t"], voltage_v, time_s)
        corr_rows.append({"particle_id": ep.particle_id, "correlation_peak": peak, "best_lag_timestep": lag_t, "best_lag_hours": lag_h})
    peak_g, lag_t_g, lag_h_g = _crosscorr_lag(retained_mean_soc_t, voltage_v, time_s)
    corr_rows.append({"particle_id": "global", "correlation_peak": peak_g, "best_lag_timestep": lag_t_g, "best_lag_hours": lag_h_g})
    pd.DataFrame(corr_rows).to_csv(table_dir / "validation_correlation_summary.csv", index=False)
    figure_paths.append(_plot_crosscorr_global(time_s, voltage_v, retained_mean_soc_t, out_dir))

    # 3. Segment fits (global = retained mean SOC)
    seg_rows = _segment_fits(time_s, voltage_v, retained_mean_soc_t, drive_sign)
    if seg_rows:
        pd.DataFrame(seg_rows).to_csv(table_dir / "validation_segment_fits.csv", index=False)

    # 4. Turning points
    volt_events = _turning_points(voltage_v, time_s)
    soc_events = _turning_points(retained_mean_soc_t, time_s)
    align_rows = _align_turning_points(volt_events, soc_events)
    if align_rows:
        pd.DataFrame(align_rows).to_csv(table_dir / "validation_turning_points.csv", index=False)

    # 5. Frequency (raw and inverted)
    period_v, cent_v = _fft_metrics(time_s, voltage_v)
    period_s, cent_s = _fft_metrics(time_s, retained_mean_soc_t)
    spectrum_path = _plot_spectrum(time_s, voltage_v, retained_mean_soc_t, out_dir)
    if spectrum_path:
        figure_paths.append(spectrum_path)
    spectrum_inv_path = _plot_spectrum_inverted(time_s, v_inv_norm, retained_mean_soc_t, out_dir)
    if spectrum_inv_path:
        figure_paths.append(spectrum_inv_path)

    # 6. Voltage vs SOC curves (raw and inverted)
    figure_paths.extend(_plot_voltage_vs_soc(time_s, voltage_v, retained_ep, retained_pp, retained_mean_soc_t, drive_sign, out_dir))
    inv_vs_soc = _plot_inverted_voltage_vs_soc_global(time_s, voltage_v, retained_mean_soc_t, out_dir)
    if inv_vs_soc:
        figure_paths.append(inv_vs_soc)

    # 7. Summary table (optional r2_linear_voltage_soc: V = a*SOC + b over full run)
    mean_offset = np.nanmean([r["offset_hours"] for r in align_rows if np.isfinite(r.get("offset_hours", np.nan))]) if align_rows else np.nan
    v_slopes = [r["voltage_slope"] for r in seg_rows if np.isfinite(r.get("voltage_slope", np.nan))]
    s_slopes = [r["soc_slope"] for r in seg_rows if np.isfinite(r.get("soc_slope", np.nan))]
    summary_rows = []
    for r in corr_rows:
        pid = r["particle_id"]
        if pid == "global":
            soc_signal = retained_mean_soc_t
        else:
            soc_signal = next((p["mean_soc_t"] for ep, p in zip(retained_ep, retained_pp) if ep.particle_id == pid), np.array([]))
        _, _, r2_lin = _fit_linear_r2(soc_signal, voltage_v) if len(soc_signal) == len(voltage_v) and len(soc_signal) > 1 else (np.nan, np.nan, np.nan)
        row = {
            "particle_id": pid,
            "correlation_peak": r["correlation_peak"],
            "best_lag_timestep": r["best_lag_timestep"],
            "best_lag_hours": r["best_lag_hours"],
            "voltage_segment_slopes": ";".join(map(str, v_slopes)) if pid == "global" and v_slopes else "",
            "soc_segment_slopes": ";".join(map(str, s_slopes)) if pid == "global" and s_slopes else "",
            "turning_point_mean_offset_hours": mean_offset if pid == "global" else np.nan,
            "dominant_period_h_voltage": period_v if pid == "global" else np.nan,
            "dominant_period_h_soc": period_s if pid == "global" else np.nan,
            "r2_linear_voltage_soc": r2_lin,
        }
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(table_dir / "validation_summary.csv", index=False)

    # 8. Inversion-aware segment-wise lag (per particle and global)
    segments = _segment_bounds(drive_sign)
    seg_lag_rows: List[Dict[str, Any]] = []
    for ep, p in zip(retained_ep, retained_pp):
        soc = p["mean_soc_t"]
        rows = _segmentwise_crosscorr(soc, v_inv_norm, time_s, segments)
        for r in rows:
            r["particle_id"] = ep.particle_id
        seg_lag_rows.extend(rows)
    # Global SOC
    rows_g = _segmentwise_crosscorr(retained_mean_soc_t, v_inv_norm, time_s, segments)
    for r in rows_g:
        r["particle_id"] = "global"
    seg_lag_rows.extend(rows_g)
    if seg_lag_rows:
        pd.DataFrame(seg_lag_rows).to_csv(table_dir / "segment_lag_summary.csv", index=False)

    # 9. Inversion-aware segment-wise linear fits (inverted voltage vs SOC)
    fit_rows: List[Dict[str, Any]] = []
    def _segment_fit_for_soc(label: Any, soc_series: np.ndarray) -> None:
        for seg_id, (start, end, sign) in enumerate(segments):
            if end - start < 2:
                continue
            t_seg = time_s[start:end]
            v_seg = v_inv_norm[start:end]
            s_seg = soc_series[start:end]
            v_slope, _, v_r2 = _fit_linear_r2(t_seg, v_seg)
            s_slope, _, s_r2 = _fit_linear_r2(t_seg, s_seg)
            if not (np.isfinite(v_slope) and np.isfinite(s_slope)):
                slope_ratio = np.nan
            else:
                slope_ratio = s_slope / (v_slope + 1e-12)
            fit_rows.append(
                {
                    "particle_id": label,
                    "segment_id": seg_id,
                    "segment_start_timestep": int(start),
                    "segment_end_timestep": int(end),
                    "segment_type": "rising" if sign == 1 else "falling",
                    "inverted_voltage_slope": v_slope,
                    "soc_slope": s_slope,
                    "inverted_voltage_r2": v_r2,
                    "soc_r2": s_r2,
                    "slope_ratio_or_difference": slope_ratio,
                }
            )

    for ep, p in zip(retained_ep, retained_pp):
        _segment_fit_for_soc(ep.particle_id, p["mean_soc_t"])
    _segment_fit_for_soc("global", retained_mean_soc_t)
    if fit_rows:
        pd.DataFrame(fit_rows).to_csv(table_dir / "segment_fit_comparison.csv", index=False)

    # 10. Inversion-aware turning-point alignment (per particle and global)
    tp_rows: List[Dict[str, Any]] = []
    for ep, p in zip(retained_ep, retained_pp):
        soc = p["mean_soc_t"]
        rows = _inversion_aware_turning_points(voltage_v, soc, time_s)
        for r in rows:
            r["particle_id"] = ep.particle_id
        tp_rows.extend(rows)
    rows_g2 = _inversion_aware_turning_points(voltage_v, retained_mean_soc_t, time_s)
    for r in rows_g2:
        r["particle_id"] = "global"
    tp_rows.extend(rows_g2)
    if tp_rows:
        pd.DataFrame(tp_rows).to_csv(table_dir / "turning_point_alignment.csv", index=False)

    # 11. Proxy validation summary (aggregate)
    summary_proxy: List[Dict[str, Any]] = []
    particle_ids = {r["particle_id"] for r in tp_rows} | {r["particle_id"] for r in seg_lag_rows}
    # FFT metrics for inverted voltage and SOC (global)
    dom_v_inv, _ = _fft_metrics(time_s, v_inv_norm)
    dom_soc, _ = _fft_metrics(time_s, retained_mean_soc_t)
    for pid in particle_ids:
        tp_pid = [r for r in tp_rows if r["particle_id"] == pid and np.isfinite(r.get("offset_timestep", np.nan))]
        lag_pid = [r for r in seg_lag_rows if r["particle_id"] == pid and np.isfinite(r.get("best_lag_timestep", np.nan))]
        if tp_pid:
            mean_abs_tp_step = float(np.nanmean([abs(r["offset_timestep"]) for r in tp_pid]))
            mean_abs_tp_h = float(np.nanmean([abs(r["offset_hours"]) for r in tp_pid]))
        else:
            mean_abs_tp_step = np.nan
            mean_abs_tp_h = np.nan
        if lag_pid:
            mean_seg_lag_step = float(np.nanmean([abs(r["best_lag_timestep"]) for r in lag_pid]))
            mean_seg_lag_h = float(np.nanmean([abs(r["best_lag_hours"]) for r in lag_pid]))
        else:
            mean_seg_lag_step = np.nan
            mean_seg_lag_h = np.nan
        notes = "small segment-wise lag, matched extrema" if (
            (mean_seg_lag_h is not np.nan and mean_seg_lag_h < 0.1)
        ) else ""
        summary_proxy.append(
            {
                "particle_id": pid,
                "mean_abs_turning_point_offset_timestep": mean_abs_tp_step,
                "mean_abs_turning_point_offset_hours": mean_abs_tp_h,
                "mean_segment_lag_timestep": mean_seg_lag_step,
                "mean_segment_lag_hours": mean_seg_lag_h,
                "dominant_period_voltage_h": dom_v_inv if pid == "global" else np.nan,
                "dominant_period_soc_h": dom_soc if pid == "global" else np.nan,
                "overall_validation_notes": notes,
            }
        )
    if summary_proxy:
        pd.DataFrame(summary_proxy).to_csv(table_dir / "proxy_validation_summary.csv", index=False)

    table_paths = [
        str(table_dir / "validation_correlation_summary.csv"),
        str(table_dir / "validation_summary.csv"),
    ]
    if seg_rows:
        table_paths.append(str(table_dir / "validation_segment_fits.csv"))
    if align_rows:
        table_paths.append(str(table_dir / "validation_turning_points.csv"))
    if seg_lag_rows:
        table_paths.append(str(table_dir / "segment_lag_summary.csv"))
    if fit_rows:
        table_paths.append(str(table_dir / "segment_fit_comparison.csv"))
    if tp_rows:
        table_paths.append(str(table_dir / "turning_point_alignment.csv"))
    if summary_proxy:
        table_paths.append(str(table_dir / "proxy_validation_summary.csv"))

    return figure_paths, table_paths
