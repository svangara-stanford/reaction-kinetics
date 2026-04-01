"""
Orchestrate: load SOC, save full-field artifact, load masks, build episodes,
geometry, rate, descriptive, plotting, save tables and pixel_observations.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from reaction_kinetics.alignment import build_electrochem_trace
from reaction_kinetics.config import (
    BOUNDARY_ANALYSIS_EXCLUDE_IDS,
    BOUNDARY_FIT_SOC_GRID_SIZE,
    DEFAULT_ERODE_BOUNDARY_PX,
    DEFAULT_SMOOTH_WINDOW,
    FIGURES_DIR,
    FULL_FIELD_SOC_FILENAME,
    I_TOT_A,
    PIXEL_AREA_CM2,
    get_data_root,
    get_output_root,
    INTERMEDIATE_DIR,
    PARTICLE_IDS_INCLUDE,
    TABLES_DIR,
)
from reaction_kinetics.current_maps import compute_scan_region_current_maps
from reaction_kinetics.current_share import (
    count_owned_pixels,
    exclusive_pixel_owner_from_masks,
    mean_weight_and_j_over_owned_pixels,
    pairwise_mask_overlap_counts,
    per_particle_share_timeseries,
)
from reaction_kinetics.segments import compute_cycle_segment_labels
from reaction_kinetics.support_audit import save_support_audit_tables
from reaction_kinetics.descriptive import compute_descriptive_summary
from reaction_kinetics.geometry import build_particle_geometry
from reaction_kinetics.io import load_a1g_movie
from reaction_kinetics.masks import (
    discover_particle_ids,
    load_particle_masks,
)
from reaction_kinetics.rate import compute_dc_dt, compute_dx_li_dt
from reaction_kinetics.schema import (
    ElectrochemTrace,
    ParticleEpisode,
    ParticleGeometry,
    RateMaps,
    DescriptiveSummary,
)
from reaction_kinetics.soc import (
    build_soc_movie,
    charged_fraction_proxy_from_heights,
    x_li_from_charged_fraction,
)


def _crop_and_mask(
    soc_full_tyx: np.ndarray,
    masks_tyx: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pad: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Crop full-frame (T,Y,X) to bbox with padding. Apply per-timestep mask. Return (soc_crop, valid_crop, origin)."""
    x_min, y_min, x_max, y_max = bbox
    ny_full, nx_full = soc_full_tyx.shape[1], soc_full_tyx.shape[2]
    y0 = max(0, y_min - pad)
    y1 = min(ny_full, y_max + 1 + pad)
    x0 = max(0, x_min - pad)
    x1 = min(nx_full, x_max + 1 + pad)
    origin = (x0, y0)
    soc_crop = soc_full_tyx[:, y0:y1, x0:x1].copy()
    mask_crop = masks_tyx[:, y0:y1, x0:x1].copy()
    valid = mask_crop & np.isfinite(soc_crop)
    soc_crop[~mask_crop] = np.nan
    return soc_crop, valid, origin


def _crop_movie(
    movie_tyx: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pad: int = 0,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop arbitrary (T,Y,X) movie to bbox with padding; returns (crop, origin)."""
    x_min, y_min, x_max, y_max = bbox
    ny_full, nx_full = movie_tyx.shape[1], movie_tyx.shape[2]
    y0 = max(0, y_min - pad)
    y1 = min(ny_full, y_max + 1 + pad)
    x0 = max(0, x_min - pad)
    x1 = min(nx_full, x_max + 1 + pad)
    origin = (x0, y0)
    return movie_tyx[:, y0:y1, x0:x1].copy(), origin


def build_episode(
    particle_id: int,
    soc_full_tyx: np.ndarray,
    masks_tyx: np.ndarray,
    mask_union_xy: np.ndarray,
    mask_intersection_xy: Optional[np.ndarray],
    trace: ElectrochemTrace,
    crop_pad: int = 0,
    erode_boundary_px: int = 0,
) -> ParticleEpisode:
    """Build one ParticleEpisode: crop to bbox, per-timestep mask, align voltage."""
    geometry = build_particle_geometry(
        particle_id,
        mask_union_xy,
        mask_intersection_xy,
        erosion_for_interior_px=erode_boundary_px,
    )
    bbox = geometry.bbox
    soc_crop, valid_crop, origin = _crop_and_mask(soc_full_tyx, masks_tyx, bbox, crop_pad)
    # Per-timestep mask area and overlap for coverage metrics
    x_min, y_min, x_max, y_max = bbox
    ny_full, nx_full = masks_tyx.shape[1], masks_tyx.shape[2]
    y0 = max(0, y_min - crop_pad)
    y1 = min(ny_full, y_max + 1 + crop_pad)
    x0 = max(0, x_min - crop_pad)
    x1 = min(nx_full, x_max + 1 + crop_pad)
    mask_crop = masks_tyx[:, y0:y1, x0:x1]
    mask_area_t = mask_crop.sum(axis=(1, 2)).astype(np.int64)
    overlap_area_t = np.array(
        [(mask_crop[t] & mask_crop[t + 1]).sum() for t in range(mask_crop.shape[0] - 1)],
        dtype=np.int64,
    )
    metadata = {"mask_area_t": mask_area_t, "overlap_area_t": overlap_area_t}
    # Align trace to same T
    T = soc_crop.shape[0]
    if len(trace.time_s) != T:
        raise ValueError(f"Trace length {len(trace.time_s)} != SOC T {T}")
    return ParticleEpisode(
        episode_id=f"particle_{particle_id}",
        particle_id=particle_id,
        time_s=trace.time_s,
        timestep_index=trace.timestep.astype(int),
        soc_movie_tyx=soc_crop,
        valid_mask_tyx=valid_crop,
        voltage_v=trace.voltage_v,
        drive_sign=trace.drive_sign,
        drive_mode=trace.drive_mode,
        geometry=geometry,
        metadata=metadata,
        crop_origin_xy=origin,
    )


def run(
    data_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    smooth_window: Optional[int] = DEFAULT_SMOOTH_WINDOW,
    erode_boundary_px: int = DEFAULT_ERODE_BOUNDARY_PX,
    crop_pad: int = 1,
    particle_ids_include: Optional[List[int]] = None,
    analysis_field: str = "soc",
) -> DescriptiveSummary:
    """
    Full pipeline: load data, build SOC movie, save full-field artifact, load masks,
    build episodes, compute rates and descriptive metrics, call plotting, save tables.
    """
    data_root = data_root or get_data_root()
    output_root = output_root or get_output_root()
    data_root = Path(data_root)
    output_root = Path(output_root)

    # Load A1g and build SOC + physically grounded stoichiometry movie
    a1g_c, a1g_d, timesteps, (ny, nx) = load_a1g_movie(data_root)
    soc_full_tyx, _ = build_soc_movie(a1g_c, a1g_d)
    s_full_tyx = charged_fraction_proxy_from_heights(a1g_c, a1g_d)
    x_li_full_tyx = x_li_from_charged_fraction(s_full_tyx)

    # Save full-field SOC for debugging
    out_intermediate = output_root / INTERMEDIATE_DIR
    out_intermediate.mkdir(parents=True, exist_ok=True)
    np.save(out_intermediate / FULL_FIELD_SOC_FILENAME, soc_full_tyx)
    np.save(out_intermediate / "s_movie_full_field_tyx_T.npy", s_full_tyx)
    np.save(out_intermediate / "x_li_movie_full_field_tyx_T.npy", x_li_full_tyx)

    # Alignment
    trace = build_electrochem_trace(data_root, timesteps)
    segment_label_t = compute_cycle_segment_labels(trace.drive_sign)

    # Full-field x_li rate (T-1 midpoint aligned) and scan-region-wide current partition
    x_li_rate_full = compute_dx_li_dt(
        x_li_full_tyx,
        trace.time_s,
        valid_mask_tyx=None,
        smooth_window=smooth_window,
        erode_boundary_px=0,
    )
    dx_li_dt_full_tyx = x_li_rate_full.dc_dt_tyx
    np.save(out_intermediate / "dx_li_dt_full_field_tyx_Tminus1.npy", dx_li_dt_full_tyx)

    scan_region_maps = compute_scan_region_current_maps(
        dx_li_dt_full_tyx,
        i_tot_a=I_TOT_A,
        pixel_area_cm2=PIXEL_AREA_CM2,
    )
    np.save(
        out_intermediate / "relative_current_weight_full_field_tyx_Tminus1.npy",
        scan_region_maps["relative_current_weight_tyx"],
    )
    np.save(
        out_intermediate / "scan_region_allocated_current_a_full_field_tyx_Tminus1.npy",
        scan_region_maps["scan_region_allocated_current_a_tyx"],
    )
    np.save(
        out_intermediate
        / "scan_region_normalized_current_density_proxy_a_per_cm2_full_field_tyx_Tminus1.npy",
        scan_region_maps["scan_region_normalized_current_density_proxy_a_per_cm2_tyx"],
    )

    # Particles (optionally filter to PARTICLE_IDS_INCLUDE or particle_ids_include)
    particle_ids = discover_particle_ids(data_root)
    filter_ids = particle_ids_include if particle_ids_include is not None else PARTICLE_IDS_INCLUDE
    if filter_ids is not None:
        particle_ids = sorted([p for p in particle_ids if p in filter_ids])
    if not particle_ids:
        return DescriptiveSummary(per_particle=[], global_metrics={})

    episodes: List[ParticleEpisode] = []
    rate_maps_list: List[RateMaps] = []
    mask_qa: Dict[int, Dict[str, Any]] = {}
    mask_union_by_id: Dict[int, np.ndarray] = {}

    for pid in particle_ids:
        masks_tyx, mask_union, mask_inter, occupancy_xy, mask_meta = load_particle_masks(
            data_root, pid, ny, nx
        )
        mask_union_by_id[pid] = mask_union
        mask_qa[pid] = {
            "mask_union_xy": mask_union,
            "mask_intersection_xy": mask_inter,
            "occupancy_xy": occupancy_xy,
            "metadata": mask_meta,
        }
        ep = build_episode(
            pid,
            soc_full_tyx,
            masks_tyx,
            mask_union,
            mask_inter,
            trace,
            crop_pad=crop_pad,
            erode_boundary_px=erode_boundary_px,
        )
        ep.metadata["segment_label_t"] = segment_label_t
        bbox = ep.geometry.bbox
        s_crop, _ = _crop_movie(s_full_tyx, bbox, crop_pad)
        x_li_crop, _ = _crop_movie(x_li_full_tyx, bbox, crop_pad)
        dx_li_dt_crop, _ = _crop_movie(dx_li_dt_full_tyx, bbox, crop_pad)
        rel_w_crop, _ = _crop_movie(scan_region_maps["relative_current_weight_tyx"], bbox, crop_pad)
        allocated_i_crop, _ = _crop_movie(
            scan_region_maps["scan_region_allocated_current_a_tyx"], bbox, crop_pad
        )
        j_crop, _ = _crop_movie(
            scan_region_maps["scan_region_normalized_current_density_proxy_a_per_cm2_tyx"],
            bbox,
            crop_pad,
        )
        ep.metadata["s_movie_tyx_T"] = s_crop
        ep.metadata["x_li_movie_tyx_T"] = x_li_crop
        ep.metadata["dx_li_dt_tyx_Tminus1"] = dx_li_dt_crop
        ep.metadata["relative_current_weight_tyx_Tminus1"] = rel_w_crop
        ep.metadata["scan_region_allocated_current_a_tyx_Tminus1"] = allocated_i_crop
        ep.metadata["scan_region_normalized_current_density_proxy_a_per_cm2_tyx_Tminus1"] = (
            j_crop
        )
        episodes.append(ep)
        valid_both = ep.valid_mask_tyx[:-1] & ep.valid_mask_tyx[1:]  # T-1
        rate_maps_list.append(
            compute_dc_dt(
                ep.soc_movie_tyx,
                ep.time_s,
                valid_mask_tyx=valid_both,
                smooth_window=smooth_window,
                erode_boundary_px=erode_boundary_px,
            )
        )

    # Exclusive pixel ownership (lowest particle_id wins) for current-share aggregation
    pixel_owner = exclusive_pixel_owner_from_masks(particle_ids, mask_union_by_id)
    share_by_pid = per_particle_share_timeseries(
        scan_region_maps["relative_current_weight_tyx"],
        pixel_owner,
        particle_ids,
        scan_region_maps["partition_invalid_t"],
    )
    union_tracked_yx = np.zeros((ny, nx), dtype=bool)
    for pid in particle_ids:
        union_tracked_yx |= mask_union_by_id[pid].astype(bool)
    scan_region_maps_tracked = compute_scan_region_current_maps(
        dx_li_dt_full_tyx,
        i_tot_a=I_TOT_A,
        pixel_area_cm2=PIXEL_AREA_CM2,
        support_mask_yx=union_tracked_yx,
    )
    share_by_pid_tracked = per_particle_share_timeseries(
        scan_region_maps_tracked["relative_current_weight_tyx"],
        pixel_owner,
        particle_ids,
        scan_region_maps_tracked["partition_invalid_t"],
    )
    for ep in episodes:
        ep.metadata["particle_current_share_Tminus1"] = share_by_pid.get(ep.particle_id)

    # Save mask QA
    for pid, qa in mask_qa.items():
        np.savez_compressed(
            out_intermediate / f"mask_qa_particle_{pid}.npz",
            union=qa["mask_union_xy"],
            intersection=qa["mask_intersection_xy"],
            occupancy=qa["occupancy_xy"],
        )

    # Optional analysis switch: preserve default SOC behavior, allow x_li-based summaries.
    analysis_field_norm = str(analysis_field).strip().lower()
    analysis_episodes = episodes
    analysis_rate_maps_list = rate_maps_list
    if analysis_field_norm == "x_li":
        analysis_episodes = []
        analysis_rate_maps_list = []
        for ep in episodes:
            x_li = ep.metadata.get("x_li_movie_tyx_T")
            dx_li_dt = ep.metadata.get("dx_li_dt_tyx_Tminus1")
            if x_li is None or dx_li_dt is None:
                continue
            ep_alt = deepcopy(ep)
            ep_alt.soc_movie_tyx = x_li
            analysis_episodes.append(ep_alt)
            analysis_rate_maps_list.append(
                RateMaps(
                    dc_dt_tyx=dx_li_dt,
                    time_mid_s=ep.time_s[:-1] * 0.5 + ep.time_s[1:] * 0.5,
                    valid_mask_tyx=np.isfinite(dx_li_dt),
                    dt_s=np.diff(ep.time_s),
                    smoothing_metadata={"analysis_field": "x_li"},
                )
            )

    # Descriptive summary (plotting will add paths)
    summary = compute_descriptive_summary(analysis_episodes, analysis_rate_maps_list)

    # Plotting
    out_figures = output_root / FIGURES_DIR
    out_figures.mkdir(parents=True, exist_ok=True)
    try:
        from reaction_kinetics.plotting import (
            plot_additional_xli_current_maps,
            plot_current_share_xli_diagnostics,
            plot_all,
        )
        figure_paths = plot_all(
            analysis_episodes,
            analysis_rate_maps_list,
            summary,
            str(out_figures),
            analysis_field=analysis_field_norm,
        )
        figure_paths.extend(
            plot_additional_xli_current_maps(
                episodes,
                x_li_rate_full.time_mid_s,
                scan_region_maps,
                str(out_figures),
            )
        )
        figure_paths.extend(
            plot_current_share_xli_diagnostics(
                particle_ids,
                mask_union_by_id,
                share_by_pid,
                scan_region_maps,
                x_li_rate_full.time_mid_s,
                str(out_figures),
            )
        )
        summary.figure_paths = figure_paths
    except Exception as e:
        summary.figure_paths = summary.figure_paths or []
        pass  # plotting optional; continue to tables

    out_tables = output_root / TABLES_DIR
    out_tables.mkdir(parents=True, exist_ok=True)

    # Boundary-only kinetics (particles 2, 4, 5, 6, 7)
    try:
        bfig_paths, btab_paths = _run_boundary_only_analysis(
            analysis_episodes, analysis_rate_maps_list, out_figures, out_tables
        )
        summary.figure_paths = list(summary.figure_paths) + bfig_paths
        summary.table_paths = list(summary.table_paths) + btab_paths
    except Exception:
        pass

    # Voltage–SOC proxy validation (retained particles only)
    try:
        if analysis_field_norm != "x_li":
            from reaction_kinetics.voltage_soc_validation import run_voltage_soc_validation

            vfig_paths, vtab_paths = run_voltage_soc_validation(
                episodes, summary, out_figures, out_tables, BOUNDARY_ANALYSIS_EXCLUDE_IDS
            )
            summary.figure_paths = list(summary.figure_paths) + vfig_paths
            summary.table_paths = list(summary.table_paths) + vtab_paths
    except Exception:
        pass

    table_paths = _save_tables(
        episodes,
        rate_maps_list,
        summary,
        out_tables,
        scan_region_maps,
        x_li_rate_full.time_mid_s,
        particle_ids,
        mask_union_by_id,
        pixel_owner,
        share_by_pid,
        scan_region_maps_tracked,
        share_by_pid_tracked,
        union_tracked_yx,
        dx_li_dt_full_tyx,
    )
    summary.table_paths = list(summary.table_paths) + table_paths

    try:
        sup_paths, _, w_df, a_df, b_df = save_support_audit_tables(
            out_tables,
            dx_li_dt_full_tyx,
            pixel_owner,
            particle_ids,
            union_tracked_yx,
            scan_region_maps,
            x_li_rate_full.time_mid_s,
            episodes,
        )
        summary.table_paths = list(summary.table_paths) + sup_paths
        if w_df is not None and a_df is not None and b_df is not None:
            from reaction_kinetics.plotting import (
                plot_area_vs_weight_comparison,
                plot_boundary_weight_vs_interior,
                plot_weight_region_breakdown_vs_time,
            )

            od = str(out_figures)
            summary.figure_paths = list(summary.figure_paths) + [
                plot_weight_region_breakdown_vs_time(x_li_rate_full.time_mid_s, w_df, od),
                plot_area_vs_weight_comparison(x_li_rate_full.time_mid_s, a_df, od),
                plot_boundary_weight_vs_interior(b_df, od),
            ]
    except Exception:
        pass

    return summary


def _run_boundary_only_analysis(
    episodes: List[ParticleEpisode],
    rate_maps_list: List[RateMaps],
    out_figures: Path,
    out_tables: Path,
) -> Tuple[List[str], List[str]]:
    """Boundary-only kinetics for retained particles (exclude 1, 3, 8). Returns (figure_paths, table_paths)."""
    from reaction_kinetics.boundary_kinetics import (
        collect_boundary_only_kinetics,
        collect_full_particle_pooled,
        run_boundary_fits,
    )
    from reaction_kinetics.plotting import (
        plot_boundary_only_dc_dt_vs_soc_per_particle,
        plot_boundary_only_pooled,
        plot_boundary_vs_full_fits,
    )
    exclude = BOUNDARY_ANALYSIS_EXCLUDE_IDS
    per_particle, pooled_norm, pooled_raw = collect_boundary_only_kinetics(
        episodes, rate_maps_list, exclude
    )
    if not per_particle:
        return [], []
    full_pooled = collect_full_particle_pooled(episodes, rate_maps_list, exclude)
    soc_grid = np.linspace(0, 1, BOUNDARY_FIT_SOC_GRID_SIZE)
    boundary_fits, full_fits, fit_quality = run_boundary_fits(
        soc_grid, pooled_raw, full_pooled
    )
    fig_paths: List[str] = []
    fig_paths.extend(
        plot_boundary_only_dc_dt_vs_soc_per_particle(per_particle, out_figures)
    )
    fig_paths.extend(plot_boundary_only_pooled(pooled_norm, out_figures))
    fig_paths.append(
        plot_boundary_vs_full_fits(soc_grid, boundary_fits, full_fits, out_figures)
    )
    tab_paths = _save_boundary_tables(
        per_particle, soc_grid, boundary_fits, full_fits, fit_quality, out_tables
    )
    return fig_paths, tab_paths


def _save_boundary_tables(
    per_particle: Dict[int, Any],
    soc_grid: np.ndarray,
    boundary_fits: Dict[str, Dict[str, np.ndarray]],
    full_fits: Dict[str, Dict[str, np.ndarray]],
    fit_quality: Dict[str, Dict[str, float]],
    out_tables: Path,
) -> List[str]:
    """Write boundary_observations_summary.csv, boundary_fitted_curves.csv, boundary_fit_quality.csv."""
    paths = []
    rows = []
    for pid, data in per_particle.items():
        for seg in ("rising", "falling"):
            n_obs = data["n_rising"] if seg == "rising" else data["n_falling"]
            mean_rate = (
                data["mean_abs_rate_rising"] if seg == "rising" else data["mean_abs_rate_falling"]
            )
            rows.append({
                "particle_id": pid,
                "segment": seg,
                "n_observations": n_obs,
                "mean_boundary_rate_magnitude": mean_rate,
            })
    if rows:
        pd.DataFrame(rows).to_csv(out_tables / "boundary_observations_summary.csv", index=False)
        paths.append(str(out_tables / "boundary_observations_summary.csv"))

    curve_rows = []
    for i, soc in enumerate(soc_grid):
        for seg in ("rising", "falling"):
            for rate_type in ("signed", "abs"):
                bv = boundary_fits.get(seg, {}).get(rate_type)
                fv = full_fits.get(seg, {}).get(rate_type)
                curve_rows.append({
                    "soc_grid": soc,
                    "segment": seg,
                    "rate_type": rate_type,
                    "boundary_fitted_value": bv[i] if bv is not None and i < len(bv) else np.nan,
                    "full_particle_fitted_value": fv[i] if fv is not None and i < len(fv) else np.nan,
                })
    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(out_tables / "boundary_fitted_curves.csv", index=False)
        paths.append(str(out_tables / "boundary_fitted_curves.csv"))

    quality_rows = []
    for seg in ("rising", "falling"):
        for rate_type in ("signed", "abs"):
            q = fit_quality.get(seg, {}).get(rate_type, np.nan)
            quality_rows.append({"segment": seg, "rate_type": rate_type, "variance_explained": q})
    if quality_rows:
        pd.DataFrame(quality_rows).to_csv(out_tables / "boundary_fit_quality.csv", index=False)
        paths.append(str(out_tables / "boundary_fit_quality.csv"))
    return paths


def _save_tables(
    episodes: List[ParticleEpisode],
    rate_maps_list: List[RateMaps],
    summary: DescriptiveSummary,
    out_tables: Path,
    scan_region_maps: Dict[str, np.ndarray],
    time_mid_s: np.ndarray,
    particle_ids_run: List[int],
    mask_union_by_id: Dict[int, np.ndarray],
    pixel_owner: np.ndarray,
    share_by_pid: Dict[int, np.ndarray],
    scan_region_maps_tracked: Dict[str, np.ndarray],
    share_by_pid_tracked: Dict[int, np.ndarray],
    union_tracked_yx: np.ndarray,
    dx_li_dt_full_tyx: np.ndarray,
) -> List[str]:
    """Save particle_summary, global_summary, episode_metadata, pixel_observations."""
    paths = []

    # particle_summary.csv
    rows = []
    for p in summary.per_particle:
        rows.append({
            "particle_id": p["particle_id"],
            "onset_soc_timestep": p.get("onset_soc_timestep"),
            "onset_rate_timestep": p.get("onset_rate_timestep"),
            "peak_rate_timestep": p.get("peak_rate_timestep"),
            "union_area": p.get("union_area"),
            "mean_mask_area": p.get("mean_mask_area"),
            "mean_occupancy_fraction": p.get("mean_occupancy_fraction"),
            "mean_conditional_soc_validity": p.get("mean_conditional_soc_validity"),
        })
    pd.DataFrame(rows).to_csv(out_tables / "particle_summary.csv", index=False)
    paths.append(str(out_tables / "particle_summary.csv"))

    # boundary_vs_interior_summary.csv
    bvi_rows = []
    for p in summary.per_particle:
        bvi_rows.append({
            "particle_id": p["particle_id"],
            "mean_soc_boundary": p.get("mean_soc_boundary_global"),
            "mean_soc_interior": p.get("mean_soc_interior_global"),
            "mean_abs_rate_boundary": p.get("mean_abs_rate_boundary_global"),
            "mean_abs_rate_interior": p.get("mean_abs_rate_interior_global"),
        })
    if bvi_rows:
        pd.DataFrame(bvi_rows).to_csv(out_tables / "boundary_vs_interior_summary.csv", index=False)
        paths.append(str(out_tables / "boundary_vs_interior_summary.csv"))

    # segment_summary.csv
    seg_rows = []
    for p in summary.per_particle:
        for seg_name in ("early_rising", "late_rising", "early_falling", "late_falling"):
            sh = (p.get("segment_heterogeneity") or {}).get(seg_name)
            sm = (p.get("segment_mean_abs_rate") or {}).get(seg_name)
            sn = (p.get("segment_n_timesteps") or {}).get(seg_name)
            seg_rows.append({
                "particle_id": p["particle_id"],
                "segment": seg_name,
                "mean_std_soc": sh,
                "mean_mean_abs_rate": sm,
                "n_timesteps": sn,
            })
    if seg_rows:
        pd.DataFrame(seg_rows).to_csv(out_tables / "segment_summary.csv", index=False)
        paths.append(str(out_tables / "segment_summary.csv"))

    # per_particle_diagnostics.csv
    diag_rows = []
    for p in summary.per_particle:
        ri = p.get("mean_abs_rate_interior_global")
        rb = p.get("mean_abs_rate_boundary_global")
        if ri is not None and np.isfinite(ri) and ri > 0 and rb is not None and np.isfinite(rb):
            ratio = float(rb) / float(ri)
        else:
            ratio = np.nan
        diag_rows.append({
            "particle_id": p["particle_id"],
            "n_valid_pixels": p.get("union_area"),
            "mean_conditional_soc_validity": p.get("mean_conditional_soc_validity"),
            "mean_conditional_rate_validity": p.get("mean_conditional_rate_validity"),
            "occupancy_fraction": p.get("mean_occupancy_fraction"),
            "onset_soc_timestep": p.get("onset_soc_timestep"),
            "peak_rate_timestep": p.get("peak_rate_timestep"),
            "boundary_vs_interior_rate_ratio": ratio,
        })
    if diag_rows:
        pd.DataFrame(diag_rows).to_csv(out_tables / "per_particle_diagnostics.csv", index=False)
        paths.append(str(out_tables / "per_particle_diagnostics.csv"))

    # global_summary.csv
    gm = summary.global_metrics
    pd.DataFrame([{"key": k, "value": str(v)[:500]} for k, v in gm.items()]).to_csv(
        out_tables / "global_summary.csv", index=False
    )
    paths.append(str(out_tables / "global_summary.csv"))

    # episode_metadata.csv
    ep_rows = []
    for ep in episodes:
        ep_rows.append({
            "episode_id": ep.episode_id,
            "particle_id": ep.particle_id,
            "crop_origin_x": ep.crop_origin_xy[0],
            "crop_origin_y": ep.crop_origin_xy[1],
            "area_px": ep.geometry.area_px,
            "area_um2": ep.geometry.area_um2,
        })
    pd.DataFrame(ep_rows).to_csv(out_tables / "episode_metadata.csv", index=False)
    paths.append(str(out_tables / "episode_metadata.csv"))

    # scan-region current partition summary (T-1)
    rows_t = []
    for t in range(len(time_mid_s)):
        rows_t.append({
            "timestep_mid_index_Tminus1": int(t),
            "time_mid_s": float(time_mid_s[t]),
            "valid_pixel_count": int(scan_region_maps["valid_pixel_count_t"][t]),
            "sum_abs_dx_li_dt": float(scan_region_maps["sum_abs_dx_li_dt_t"][t]),
            "zero_denominator_frame": bool(scan_region_maps["zero_denominator_t"][t]),
            "near_zero_denominator_frame": bool(scan_region_maps["near_zero_denominator_t"][t]),
            "partition_invalid_frame": bool(scan_region_maps["partition_invalid_t"][t]),
        })
    if rows_t:
        pd.DataFrame(rows_t).to_csv(
            out_tables / "scan_region_current_partition_timestep_summary.csv", index=False
        )
        paths.append(str(out_tables / "scan_region_current_partition_timestep_summary.csv"))

    # mask_overlap_audit.csv (union masks; timestep=-1 denotes time-aggregated union)
    overlap_rows = pairwise_mask_overlap_counts(particle_ids_run, mask_union_by_id)
    if overlap_rows:
        odf = pd.DataFrame(overlap_rows)
        odf.insert(0, "timestep", -1)
        odf.to_csv(out_tables / "mask_overlap_audit.csv", index=False)
        paths.append(str(out_tables / "mask_overlap_audit.csv"))

    w_full = scan_region_maps["relative_current_weight_tyx"]
    j_full = scan_region_maps["scan_region_normalized_current_density_proxy_a_per_cm2_tyx"]
    zero_denom = scan_region_maps["zero_denominator_t"]
    near_zero_denom = scan_region_maps["near_zero_denominator_t"]
    partition_invalid = scan_region_maps["partition_invalid_t"]
    retained_n_pixels = count_owned_pixels(pixel_owner, particle_ids_run)

    # current_share_audit.csv (T-1)
    audit_rows = []
    for t in range(len(time_mid_s)):
        valid_n = int(scan_region_maps["valid_pixel_count_t"][t])
        if partition_invalid[t]:
            sum_ret = np.nan
            uncovered = np.nan
            if near_zero_denom[t]:
                note = "partition_invalid_near_zero_denominator"
            elif zero_denom[t]:
                note = "partition_invalid_zero_or_nonfinite_denominator"
            else:
                note = "partition_invalid"
        else:
            sum_ret = float(
                np.sum([share_by_pid[pid][t] for pid in particle_ids_run])
            )
            uncovered = float(1.0 - sum_ret)
            note = "exclusive_lowest_particle_id"
        audit_rows.append({
            "timestep_mid_index_Tminus1": int(t),
            "time_mid_s": float(time_mid_s[t]),
            "partition_invalid_frame": bool(partition_invalid[t]),
            "sum_retained_particle_shares": sum_ret,
            "retained_particle_pixel_count": retained_n_pixels,
            "total_valid_scanned_pixel_count": valid_n,
            "uncovered_weight_fraction": uncovered,
            "note": note,
        })
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(out_tables / "current_share_audit.csv", index=False)
        paths.append(str(out_tables / "current_share_audit.csv"))

    # Support sensitivity: full valid scan vs union of tracked particle masks only
    sens_rows = []
    for support_mode, maps, shares in (
        ("full_valid_scan", scan_region_maps, share_by_pid),
        ("tracked_particle_union_only", scan_region_maps_tracked, share_by_pid_tracked),
    ):
        pinv = maps["partition_invalid_t"]
        for t in range(len(time_mid_s)):
            if support_mode == "full_valid_scan":
                sp_count = int(maps["valid_pixel_count_t"][t])
            else:
                finite_d = np.isfinite(dx_li_dt_full_tyx[t])
                sp_count = int(np.sum(union_tracked_yx & finite_d))
            if pinv[t]:
                sum_ret_s = np.nan
                uncovered_s = np.nan
            else:
                sum_ret_s = float(
                    np.sum([shares[pid][t] for pid in particle_ids_run])
                )
                uncovered_s = float(1.0 - sum_ret_s)
            sens_rows.append({
                "support_mode": support_mode,
                "timestep_mid_index_Tminus1": int(t),
                "time_mid_s": float(time_mid_s[t]),
                "support_pixel_count": sp_count,
                "partition_invalid_frame": bool(pinv[t]),
                "sum_retained_particle_shares": sum_ret_s,
                "uncovered_weight_fraction": uncovered_s,
            })
    if sens_rows:
        pd.DataFrame(sens_rows).to_csv(
            out_tables / "current_share_support_sensitivity.csv", index=False
        )
        paths.append(str(out_tables / "current_share_support_sensitivity.csv"))

    # per-particle current share over time (T-1); exclusive-ownership sums
    share_rows = []
    agg_rows = []
    for pid in particle_ids_run:
        share_t = share_by_pid.get(pid)
        if share_t is None:
            continue
        mean_w_t, mean_j_t = mean_weight_and_j_over_owned_pixels(
            w_full, j_full, pixel_owner, pid, partition_invalid
        )
        for t in range(min(len(time_mid_s), len(share_t))):
            share_rows.append({
                "particle_id": pid,
                "timestep_mid_index_Tminus1": int(t),
                "time_mid_s": float(time_mid_s[t]),
                "particle_current_share": float(share_t[t]) if np.isfinite(share_t[t]) else np.nan,
                "mean_relative_current_weight_on_owned_pixels": (
                    float(mean_w_t[t]) if np.isfinite(mean_w_t[t]) else np.nan
                ),
                "mean_scan_region_normalized_current_density_proxy_a_per_cm2": (
                    float(mean_j_t[t]) if np.isfinite(mean_j_t[t]) else np.nan
                ),
                "zero_denominator_frame": bool(zero_denom[t]),
                "near_zero_denominator_frame": bool(near_zero_denom[t]),
                "partition_invalid_frame": bool(partition_invalid[t]),
            })
        agg_rows.append({
            "particle_id": pid,
            "mean_current_share": float(np.nanmean(share_t)) if np.any(np.isfinite(share_t)) else np.nan,
            "peak_current_share": float(np.nanmax(share_t)) if np.any(np.isfinite(share_t)) else np.nan,
            "mean_relative_current_weight": (
                float(np.nanmean(mean_w_t)) if np.any(np.isfinite(mean_w_t)) else np.nan
            ),
            "peak_relative_current_weight": (
                float(np.nanmax(mean_w_t)) if np.any(np.isfinite(mean_w_t)) else np.nan
            ),
            "mean_scan_region_normalized_current_density_proxy_a_per_cm2": (
                float(np.nanmean(mean_j_t)) if np.any(np.isfinite(mean_j_t)) else np.nan
            ),
            "peak_scan_region_normalized_current_density_proxy_a_per_cm2": (
                float(np.nanmax(mean_j_t)) if np.any(np.isfinite(mean_j_t)) else np.nan
            ),
        })
    if share_rows:
        pd.DataFrame(share_rows).to_csv(
            out_tables / "per_particle_current_share_over_time_Tminus1.csv", index=False
        )
        paths.append(str(out_tables / "per_particle_current_share_over_time_Tminus1.csv"))
    if agg_rows:
        pd.DataFrame(agg_rows).to_csv(
            out_tables / "per_particle_current_share_summary.csv", index=False
        )
        paths.append(str(out_tables / "per_particle_current_share_summary.csv"))

    # pixel_observations (long-form, full-frame x,y)
    dfs = []
    for ep, rate in zip(episodes, rate_maps_list):
        T, Hy, Wx = ep.soc_movie_tyx.shape
        x0, y0 = ep.crop_origin_xy
        s_tyx = ep.metadata.get("s_movie_tyx_T")
        x_li_tyx = ep.metadata.get("x_li_movie_tyx_T")
        dx_li_dt_tyx = ep.metadata.get("dx_li_dt_tyx_Tminus1")
        rel_w_tyx = ep.metadata.get("relative_current_weight_tyx_Tminus1")
        allocated_i_tyx = ep.metadata.get("scan_region_allocated_current_a_tyx_Tminus1")
        j_tyx = ep.metadata.get(
            "scan_region_normalized_current_density_proxy_a_per_cm2_tyx_Tminus1"
        )
        for t in range(T):
            for y in range(Hy):
                for x in range(Wx):
                    if not ep.valid_mask_tyx[t, y, x]:
                        continue
                    soc_val = ep.soc_movie_tyx[t, y, x]
                    if not np.isfinite(soc_val):
                        continue
                    x_glob = x0 + x
                    y_glob = y0 + y
                    dc_dt_val = np.nan
                    time_mid = np.nan
                    if t < T - 1 and rate.valid_mask_tyx[t, y, x]:
                        dc_dt_val = rate.dc_dt_tyx[t, y, x]
                        time_mid = rate.time_mid_s[t]
                    dx_li_dt_val = np.nan
                    rel_weight = np.nan
                    allocated_i = np.nan
                    j_val = np.nan
                    zd = np.nan
                    nzd = np.nan
                    pinv = np.nan
                    if t < T - 1:
                        if dx_li_dt_tyx is not None:
                            dx_li_dt_val = dx_li_dt_tyx[t, y, x]
                        if rel_w_tyx is not None:
                            rel_weight = rel_w_tyx[t, y, x]
                        if allocated_i_tyx is not None:
                            allocated_i = allocated_i_tyx[t, y, x]
                        if j_tyx is not None:
                            j_val = j_tyx[t, y, x]
                        zd = bool(scan_region_maps["zero_denominator_t"][t])
                        nzd = bool(scan_region_maps["near_zero_denominator_t"][t])
                        pinv = bool(scan_region_maps["partition_invalid_t"][t])
                    yg, xg = y0 + y, x0 + x
                    dist_px = np.nan
                    dist_um = np.nan
                    if yg < ep.geometry.distance_to_boundary_px.shape[0] and xg < ep.geometry.distance_to_boundary_px.shape[1]:
                        dist_px = ep.geometry.distance_to_boundary_px[yg, xg]
                        dist_um = ep.geometry.distance_to_boundary_um[yg, xg]
                    is_bound = ep.geometry.boundary_xy[yg, xg] if yg < ep.geometry.boundary_xy.shape[0] and xg < ep.geometry.boundary_xy.shape[1] else False
                    dfs.append({
                        "particle_id": ep.particle_id,
                        "timestep": ep.timestep_index[t],
                        "time_s": ep.time_s[t],
                        "x": x_glob,
                        "y": y_glob,
                        "x_local": x,
                        "y_local": y,
                        "x_um": x_glob * 0.5,
                        "y_um": y_glob * 0.5,
                        "soc": soc_val,
                        "s_proxy": s_tyx[t, y, x] if s_tyx is not None else np.nan,
                        "x_li": x_li_tyx[t, y, x] if x_li_tyx is not None else np.nan,
                        "dc_dt": dc_dt_val,
                        "dx_li_dt": dx_li_dt_val,
                        "relative_current_weight_scan_region": rel_weight,
                        "scan_region_allocated_current_a": allocated_i,
                        "scan_region_normalized_current_density_proxy_a_per_cm2": j_val,
                        "zero_denominator_frame_Tminus1": zd,
                        "near_zero_denominator_frame_Tminus1": nzd,
                        "partition_invalid_frame_Tminus1": pinv,
                        "voltage_v": ep.voltage_v[t],
                        "drive_mode": ep.drive_mode[t],
                        "drive_sign": int(ep.drive_sign[t]),
                        "distance_to_boundary_px": dist_px,
                        "distance_to_boundary_um": dist_um,
                        "is_boundary": is_bound,
                        "is_interior": not is_bound and np.isfinite(dist_px),
                    })
    if dfs:
        pdf = pd.DataFrame(dfs)
        try:
            pdf.to_parquet(out_tables / "pixel_observations.parquet", index=False)
            paths.append(str(out_tables / "pixel_observations.parquet"))
        except Exception:
            pdf.to_csv(out_tables / "pixel_observations.csv", index=False)
            paths.append(str(out_tables / "pixel_observations.csv"))

    return paths
