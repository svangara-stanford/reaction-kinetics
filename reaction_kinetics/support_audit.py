"""Support-region counts, weight allocation breakdown, and boundary vs interior weight summaries."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from reaction_kinetics.config import INTERIOR_THRESHOLD_PX
from reaction_kinetics.schema import ParticleEpisode


def _retained_mask_yx(pixel_owner: np.ndarray, particle_ids_run: List[int]) -> np.ndarray:
    ids = np.array(particle_ids_run, dtype=np.int32)
    return np.isin(pixel_owner, ids)


def compute_support_region_breakdown(
    dx_li_dt_full_tyx: np.ndarray,
    pixel_owner: np.ndarray,
    particle_ids_run: List[int],
    union_tracked_yx: np.ndarray,
    partition_invalid_t: np.ndarray,
    time_mid_s: np.ndarray,
) -> pd.DataFrame:
    """Per-timestep 3-class support counts plus outside-union alias."""
    ny, nx = pixel_owner.shape
    n_pix = ny * nx
    retained_yx = _retained_mask_yx(pixel_owner, particle_ids_run)
    union_b = union_tracked_yx.astype(bool)

    rows = []
    Tm1 = dx_li_dt_full_tyx.shape[0]
    for t in range(Tm1):
        finite_use = np.isfinite(dx_li_dt_full_tyx[t])
        n_valid = int(np.sum(finite_use))
        n_ret = int(np.sum(finite_use & retained_yx))
        n_non = int(np.sum(finite_use & ~retained_yx))
        n_inv = n_pix - n_valid
        n_out_union = int(np.sum(finite_use & ~union_b))
        if n_ret + n_non != n_valid:
            raise ValueError(f"t={t}: count mismatch retained+nonretained != valid")
        if n_out_union != n_non:
            raise ValueError(f"t={t}: outside-union count {n_out_union} != nonretained {n_non}")

        frac_ret = float(n_ret / n_valid) if n_valid > 0 else np.nan
        frac_non = float(n_non / n_valid) if n_valid > 0 else np.nan
        rows.append({
            "timestep_mid_index_Tminus1": t,
            "time_mid_s": float(time_mid_s[t]) if t < len(time_mid_s) else np.nan,
            "n_valid_total": n_valid,
            "n_retained_particle_owned": n_ret,
            "n_nonretained_valid": n_non,
            "n_invalid_or_nan": n_inv,
            "n_valid_outside_particle_union": n_out_union,
            "n_obvious_background_or_offparticle": n_out_union,
            "frac_retained_particle_owned": frac_ret,
            "frac_nonretained_valid": frac_non,
            "partition_invalid_frame": bool(partition_invalid_t[t]),
        })
    return pd.DataFrame(rows)


def compute_weight_region_breakdown(
    relative_weight_tyx: np.ndarray,
    dx_li_dt_full_tyx: np.ndarray,
    pixel_owner: np.ndarray,
    particle_ids_run: List[int],
    partition_invalid_t: np.ndarray,
    time_mid_s: np.ndarray,
) -> pd.DataFrame:
    """Per-timestep sum of relative weights on retained vs non-retained valid pixels."""
    retained_yx = _retained_mask_yx(pixel_owner, particle_ids_run)
    Tm1 = relative_weight_tyx.shape[0]
    rows = []
    for t in range(Tm1):
        finite_use = np.isfinite(dx_li_dt_full_tyx[t])
        pinv = bool(partition_invalid_t[t])
        if pinv:
            rows.append({
                "timestep_mid_index_Tminus1": t,
                "time_mid_s": float(time_mid_s[t]) if t < len(time_mid_s) else np.nan,
                "weight_on_retained_particles": np.nan,
                "weight_on_nonretained_valid_pixels": np.nan,
                "fraction_on_retained_particles": np.nan,
                "fraction_on_nonretained_valid_pixels": np.nan,
                "partition_invalid_frame": True,
            })
            continue
        w = relative_weight_tyx[t]
        sel_r = finite_use & retained_yx
        sel_n = finite_use & ~retained_yx
        wr = float(np.nansum(w[sel_r]))
        wn = float(np.nansum(w[sel_n]))
        s = wr + wn
        rows.append({
            "timestep_mid_index_Tminus1": t,
            "time_mid_s": float(time_mid_s[t]) if t < len(time_mid_s) else np.nan,
            "weight_on_retained_particles": wr,
            "weight_on_nonretained_valid_pixels": wn,
            "fraction_on_retained_particles": wr / s if s > 0 and np.isfinite(s) else np.nan,
            "fraction_on_nonretained_valid_pixels": wn / s if s > 0 and np.isfinite(s) else np.nan,
            "partition_invalid_frame": False,
        })
    return pd.DataFrame(rows)


def compute_area_vs_weight_comparison(
    support_df: pd.DataFrame,
    weight_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine support + weight tables into per-timestep area vs weight metrics."""
    rows = []
    for _, srow in support_df.iterrows():
        t = int(srow["timestep_mid_index_Tminus1"])
        wmatch = weight_df[weight_df["timestep_mid_index_Tminus1"] == t]
        if wmatch.empty:
            continue
        wrow = wmatch.iloc[0]
        if srow["partition_invalid_frame"] or wrow["partition_invalid_frame"]:
            rows.append({
                "timestep_mid_index_Tminus1": t,
                "time_mid_s": srow["time_mid_s"],
                "retained_area_fraction": np.nan,
                "retained_weight_fraction": np.nan,
                "retained_weight_per_unit_area": np.nan,
                "nonretained_weight_per_unit_area": np.nan,
                "partition_invalid_frame": True,
            })
            continue
        n_v = int(srow["n_valid_total"])
        n_r = int(srow["n_retained_particle_owned"])
        if n_v <= 0:
            raf = np.nan
        else:
            raf = float(n_r / n_v)
        rwf = float(wrow["weight_on_retained_particles"])
        if n_r > 0 and np.isfinite(raf) and raf > 0:
            rwa = rwf / raf
        else:
            rwa = np.nan
        one_m_raf = 1.0 - raf
        one_m_rwf = 1.0 - rwf
        if one_m_raf > 0 and np.isfinite(one_m_raf):
            nwpa = one_m_rwf / one_m_raf
        else:
            nwpa = np.nan
        rows.append({
            "timestep_mid_index_Tminus1": t,
            "time_mid_s": srow["time_mid_s"],
            "retained_area_fraction": raf,
            "retained_weight_fraction": rwf,
            "retained_weight_per_unit_area": rwa,
            "nonretained_weight_per_unit_area": nwpa,
            "partition_invalid_frame": False,
        })
    return pd.DataFrame(rows)


def _mean_finite(vals: List[float]) -> float:
    a = np.array(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.mean(a))


def compute_boundary_weight_summary(
    episodes: List[ParticleEpisode],
    scan_region_maps: Dict[str, np.ndarray],
    dx_li_dt_full_tyx: np.ndarray,
    particle_ids_run: List[int],
    interior_threshold_px: float = INTERIOR_THRESHOLD_PX,
) -> pd.DataFrame:
    """
    Per particle: mean over valid-partition timesteps of spatial mean relative weight
    and allocated current proxy on boundary vs interior (full-field geometry).
    """
    w_tyx = scan_region_maps["relative_current_weight_tyx"]
    alloc_tyx = scan_region_maps["scan_region_allocated_current_a_tyx"]
    pinv = scan_region_maps["partition_invalid_t"]
    Tm1 = w_tyx.shape[0]
    pid_set = set(int(x) for x in particle_ids_run)

    rows = []
    for ep in episodes:
        pid = int(ep.particle_id)
        if pid not in pid_set:
            continue
        ge = ep.geometry
        boundary_yx = ge.boundary_xy.astype(bool)
        interior_yx = (ge.distance_to_boundary_px >= interior_threshold_px) & ge.mask_union_xy

        mb_list: List[float] = []
        mi_list: List[float] = []
        ab_list: List[float] = []
        ai_list: List[float] = []

        for t in range(min(Tm1, dx_li_dt_full_tyx.shape[0])):
            if pinv[t]:
                continue
            finite_use = np.isfinite(dx_li_dt_full_tyx[t])
            wt = w_tyx[t]
            at = alloc_tyx[t]
            bsel = boundary_yx & finite_use
            isel = interior_yx & finite_use
            if np.any(bsel):
                mb_list.append(float(np.nanmean(wt[bsel])))
                ab_list.append(float(np.nanmean(at[bsel])))
            else:
                mb_list.append(float("nan"))
                ab_list.append(float("nan"))
            if np.any(isel):
                mi_list.append(float(np.nanmean(wt[isel])))
                ai_list.append(float(np.nanmean(at[isel])))
            else:
                mi_list.append(float("nan"))
                ai_list.append(float("nan"))

        mb = _mean_finite(mb_list)
        mi = _mean_finite(mi_list)
        ab = _mean_finite(ab_list)
        ai = _mean_finite(ai_list)
        if np.isfinite(mi) and mi > 0 and np.isfinite(mb):
            wratio = float(mb / mi)
        else:
            wratio = float("nan")

        rows.append({
            "particle_id": pid,
            "mean_relative_weight_boundary": mb,
            "mean_relative_weight_interior": mi,
            "boundary_vs_interior_weight_ratio": wratio,
            "mean_allocated_current_proxy_boundary": ab,
            "mean_allocated_current_proxy_interior": ai,
            "n_timesteps_valid_partition": int(np.sum(~pinv[:Tm1])),
        })
    return pd.DataFrame(rows)


def write_support_interpretation_note(
    out_path: Path,
    support_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    area_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
) -> None:
    """Write markdown summary with computed means and fixed caveats."""
    valid_w = weight_df[~weight_df["partition_invalid_frame"]].copy()
    valid_a = area_df[~area_df["partition_invalid_frame"]].copy()

    def _col_mean(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns or df.empty:
            return float("nan")
        v = pd.to_numeric(df[col], errors="coerce")
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")

    m_ra = _col_mean(valid_a, "retained_area_fraction")
    m_rw = _col_mean(valid_a, "retained_weight_fraction")
    m_rwa = _col_mean(valid_a, "retained_weight_per_unit_area")
    m_nwa = _col_mean(valid_a, "nonretained_weight_per_unit_area")
    m_frac_r = _col_mean(valid_w, "fraction_on_retained_particles")
    m_frac_n = _col_mean(valid_w, "fraction_on_nonretained_valid_pixels")

    lines = [
        "# Support interpretation (auto-generated)",
        "",
        "This note summarizes scan-region support, weight allocation, and boundary vs interior",
        "relative weights from the latest pipeline tables. It does not replace experiment-specific",
        "interpretation.",
        "",
        "## Aggregate metrics (mean over valid-partition timesteps)",
        "",
        f"- Mean retained area fraction (of valid-scan pixels): **{m_ra:.4f}**" if np.isfinite(m_ra) else "- Mean retained area fraction: *n/a*",
        f"- Mean retained weight fraction: **{m_rw:.4f}**" if np.isfinite(m_rw) else "- Mean retained weight fraction: *n/a*",
        f"- Mean fraction of weight on retained vs non-retained valid pixels: **{m_frac_r:.4f}** / **{m_frac_n:.4f}**"
        if np.isfinite(m_frac_r) and np.isfinite(m_frac_n)
        else "- Weight split: *n/a*",
        f"- Mean retained weight / retained area (uniform baseline = 1): **{m_rwa:.4f}**" if np.isfinite(m_rwa) else "- Retained weight/area: *n/a*",
        f"- Mean non-retained weight / non-retained area: **{m_nwa:.4f}**" if np.isfinite(m_nwa) else "- Non-retained weight/area: *n/a*",
        "",
        "## Interpretation hooks",
        "",
        "- If **retained weight fraction** is well below **retained area fraction**, retained",
        "  particles receive a smaller share of the scan-normalized weight than a uniform",
        "  density over valid pixels would predict — the remaining weight sits on valid pixels",
        "  outside particle masks (see `n_nonretained_valid`).",
        "- Comparing **retained_weight_per_unit_area** to **nonretained_weight_per_unit_area**",
        "  indicates whether signal outside masks is disproportionately strong (activity-like)",
        "  vs weak (closer to area-driven background), *relative to pixel count in each region*.",
        "",
        "## Boundary vs interior (relative weight)",
        "",
    ]
    if not boundary_df.empty and "boundary_vs_interior_weight_ratio" in boundary_df.columns:
        br = boundary_df["boundary_vs_interior_weight_ratio"]
        br = br[np.isfinite(br)]
        if len(br):
            lines.append(
                f"- Mean boundary/interior relative-weight ratio across particles: **{float(br.mean()):.4f}** "
                f"(individual rows in `boundary_weight_summary.csv`)."
            )
        else:
            lines.append("- Boundary/interior weight ratio: *n/a*")
    else:
        lines.append("- Boundary/interior summary not available.")

    lines.extend([
        "",
        "## What this pipeline cannot conclude (without additional metadata)",
        "",
        "- Whether **off-mask** `|dx_li_dt|` and allocated weight represent true cathode regions,",
        "  support/background, optical artifacts, or mixed effects — there is no global cathode mask",
        "  in this repository.",
        "- Absolute physical current density: scan-region weights are **normalized** by",
        "  sum(|dx_li_dt|) over the chosen support; totals use `I_TOT_A` as in the main pipeline.",
        "- Contact Donggun (or similar) for instrument geometry, mask provenance, and independent",
        "  validation of regions outside tracked particles.",
        "",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_support_audit_tables(
    out_tables: Path,
    dx_li_dt_full_tyx: np.ndarray,
    pixel_owner: np.ndarray,
    particle_ids_run: List[int],
    union_tracked_yx: np.ndarray,
    scan_region_maps: Dict[str, np.ndarray],
    time_mid_s: np.ndarray,
    episodes: List[ParticleEpisode],
):
    """
    Write support/weight/area/boundary CSVs and interpretation note.
    Returns (paths, support_df, weight_df, area_df, boundary_df).
    """
    pinv = scan_region_maps["partition_invalid_t"]
    support_df = compute_support_region_breakdown(
        dx_li_dt_full_tyx,
        pixel_owner,
        particle_ids_run,
        union_tracked_yx,
        pinv,
        time_mid_s,
    )
    weight_df = compute_weight_region_breakdown(
        scan_region_maps["relative_current_weight_tyx"],
        dx_li_dt_full_tyx,
        pixel_owner,
        particle_ids_run,
        pinv,
        time_mid_s,
    )
    area_df = compute_area_vs_weight_comparison(support_df, weight_df)
    boundary_df = compute_boundary_weight_summary(
        episodes,
        scan_region_maps,
        dx_li_dt_full_tyx,
        particle_ids_run,
    )

    paths: List[str] = []
    p1 = out_tables / "support_region_breakdown.csv"
    p2 = out_tables / "weight_region_breakdown.csv"
    p3 = out_tables / "area_vs_weight_comparison.csv"
    p4 = out_tables / "boundary_weight_summary.csv"
    p5 = out_tables / "support_interpretation_note.md"

    support_df.to_csv(p1, index=False)
    weight_df.to_csv(p2, index=False)
    area_df.to_csv(p3, index=False)
    boundary_df.to_csv(p4, index=False)
    paths.extend([str(p1), str(p2), str(p3), str(p4)])

    write_support_interpretation_note(p5, support_df, weight_df, area_df, boundary_df)
    paths.append(str(p5))

    return paths, support_df, weight_df, area_df, boundary_df
