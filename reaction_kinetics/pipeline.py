"""
Orchestrate: load SOC, save full-field artifact, load masks, build episodes,
geometry, rate, descriptive, plotting, save tables and pixel_observations.
"""

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
    get_data_root,
    get_output_root,
    INTERMEDIATE_DIR,
    PARTICLE_IDS_INCLUDE,
    TABLES_DIR,
)
from reaction_kinetics.segments import compute_cycle_segment_labels
from reaction_kinetics.descriptive import compute_descriptive_summary
from reaction_kinetics.geometry import build_particle_geometry
from reaction_kinetics.io import load_a1g_movie
from reaction_kinetics.masks import (
    discover_particle_ids,
    load_particle_masks,
)
from reaction_kinetics.rate import compute_dc_dt
from reaction_kinetics.schema import (
    ElectrochemTrace,
    ParticleEpisode,
    ParticleGeometry,
    RateMaps,
    DescriptiveSummary,
)
from reaction_kinetics.soc import build_soc_movie


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
) -> DescriptiveSummary:
    """
    Full pipeline: load data, build SOC movie, save full-field artifact, load masks,
    build episodes, compute rates and descriptive metrics, call plotting, save tables.
    """
    data_root = data_root or get_data_root()
    output_root = output_root or get_output_root()
    data_root = Path(data_root)
    output_root = Path(output_root)

    # Load A1g and build SOC
    a1g_c, a1g_d, timesteps, (ny, nx) = load_a1g_movie(data_root)
    soc_full_tyx, _ = build_soc_movie(a1g_c, a1g_d)

    # Save full-field SOC for debugging
    out_intermediate = output_root / INTERMEDIATE_DIR
    out_intermediate.mkdir(parents=True, exist_ok=True)
    np.save(out_intermediate / FULL_FIELD_SOC_FILENAME, soc_full_tyx)

    # Alignment
    trace = build_electrochem_trace(data_root, timesteps)
    segment_label_t = compute_cycle_segment_labels(trace.drive_sign)

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

    for pid in particle_ids:
        masks_tyx, mask_union, mask_inter, occupancy_xy, mask_meta = load_particle_masks(
            data_root, pid, ny, nx
        )
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

    # Save mask QA
    for pid, qa in mask_qa.items():
        np.savez_compressed(
            out_intermediate / f"mask_qa_particle_{pid}.npz",
            union=qa["mask_union_xy"],
            intersection=qa["mask_intersection_xy"],
            occupancy=qa["occupancy_xy"],
        )

    # Descriptive summary (plotting will add paths)
    summary = compute_descriptive_summary(episodes, rate_maps_list)

    # Plotting
    out_figures = output_root / FIGURES_DIR
    out_figures.mkdir(parents=True, exist_ok=True)
    try:
        from reaction_kinetics.plotting import (
            plot_all,
        )
        figure_paths = plot_all(
            episodes, rate_maps_list, summary, str(out_figures)
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
            episodes, rate_maps_list, out_figures, out_tables
        )
        summary.figure_paths = list(summary.figure_paths) + bfig_paths
        summary.table_paths = list(summary.table_paths) + btab_paths
    except Exception:
        pass

    # Voltage–SOC proxy validation (retained particles only)
    try:
        from reaction_kinetics.voltage_soc_validation import run_voltage_soc_validation
        vfig_paths, vtab_paths = run_voltage_soc_validation(
            episodes, summary, out_figures, out_tables, BOUNDARY_ANALYSIS_EXCLUDE_IDS
        )
        summary.figure_paths = list(summary.figure_paths) + vfig_paths
        summary.table_paths = list(summary.table_paths) + vtab_paths
    except Exception:
        pass

    table_paths = _save_tables(
        episodes, rate_maps_list, summary, out_tables
    )
    summary.table_paths = list(summary.table_paths) + table_paths

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

    # pixel_observations (long-form, full-frame x,y)
    dfs = []
    for ep, rate in zip(episodes, rate_maps_list):
        T, Hy, Wx = ep.soc_movie_tyx.shape
        x0, y0 = ep.crop_origin_xy
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
                        "dc_dt": dc_dt_val,
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
