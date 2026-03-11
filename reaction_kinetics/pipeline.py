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
    DEFAULT_ERODE_BOUNDARY_PX,
    DEFAULT_SMOOTH_WINDOW,
    FIGURES_DIR,
    FULL_FIELD_SOC_FILENAME,
    get_data_root,
    get_output_root,
    INTERMEDIATE_DIR,
    TABLES_DIR,
)
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

    # Particles
    particle_ids = discover_particle_ids(data_root)
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

    # Tables and pixel_observations
    out_tables = output_root / TABLES_DIR
    out_tables.mkdir(parents=True, exist_ok=True)
    table_paths = _save_tables(
        episodes, rate_maps_list, summary, out_tables
    )
    summary.table_paths = table_paths

    return summary


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
