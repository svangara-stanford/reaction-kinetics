"""Data structures for the pipeline. Arrays: movies (T,Y,X), masks (Y,X), vectors (T,)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ElectrochemTrace:
    """
    Aligned voltage and time per timestep. No current data; segments inferred from
    avg_Ewe/V vs timestep (turning points). Use voltage_trend and drive_sign_proxy
    unless charge/discharge convention is explicitly documented.
    """

    time_h: np.ndarray  # (T,) per-timestep time (e.g. mid or start)
    time_s: np.ndarray  # (T,) same in seconds
    voltage_v: np.ndarray  # (T,) avg_Ewe/V per timestep
    timestep: np.ndarray  # (T,) timestep index 0..N-1
    timestep_start_h: np.ndarray  # (T,)
    timestep_end_h: np.ndarray  # (T,)
    # Primary: trend from voltage only (no current)
    voltage_trend: np.ndarray  # (T,) "increasing" | "decreasing" | "unknown"
    drive_sign_proxy: np.ndarray  # (T,) +1 (V rising), -1 (V falling), 0 (unknown)
    # Optional labels; only set when convention is documented (see alignment.CHARGE_DISCHARGE_CONVENTION)
    drive_mode: np.ndarray  # (T,) "charge" | "discharge" | "unknown" or empty
    drive_sign: np.ndarray  # (T,) +1, -1, or 0 (same as drive_sign_proxy if convention applied)
    is_constant_current_assumed: bool = False  # always False when inferred from V only


@dataclass
class ParticleGeometry:
    """
    Geometry for one particle. All masks and arrays are (Y,X) in full-frame coordinates.
    Reference mask = union across all timesteps; use for area, centroid, boundary, bbox.
    Per-timestep masks are used for framewise SOC/rate; union/intersection for QA.
    """

    particle_id: int
    mask_xy: np.ndarray  # (Y,X) bool; reference mask = union over timesteps
    mask_union_xy: np.ndarray  # (Y,X) bool; same as mask_xy (union across timesteps)
    boundary_xy: np.ndarray  # (Y,X) bool, boundary of union mask
    area_px: int  # of union mask
    area_um2: float
    centroid_xy_px: Tuple[float, float]  # (x, y) in full-frame
    centroid_xy_um: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max) inclusive, full-frame
    distance_to_boundary_px: np.ndarray  # (Y,X) float, NaN outside mask
    distance_to_boundary_um: np.ndarray  # (Y,X)
    mask_intersection_xy: Optional[np.ndarray] = None  # (Y,X) bool; for robustness QA
    interior_xy: Optional[np.ndarray] = None  # (Y,X) bool, optional erosion
    shape_yx: Tuple[int, int] = (0, 0)  # (ny, nx) of mask_xy


@dataclass
class ParticleEpisode:
    """
    One particle's episode: SOC movie (T,Y,X) in crop, voltage, geometry.
    Coordinates: preserve both full-frame (x, y in 30x30 scan) and local crop (x_local, y_local in bbox).
    Framewise calculations use per-timestep masks; geometry uses static union (reference) mask.
    Long-form pixel table must use full-frame x, y.
    """

    episode_id: str  # e.g. "particle_1"
    particle_id: int
    time_s: np.ndarray  # (T,)
    timestep_index: np.ndarray  # (T,)
    soc_movie_tyx: np.ndarray  # (T, Y, X) SOC in crop; indices are local (y_local, x_local)
    valid_mask_tyx: np.ndarray  # (T, Y, X) bool, per-timestep valid for analysis
    voltage_v: np.ndarray  # (T,)
    drive_sign: np.ndarray  # (T,) or use drive_sign_proxy
    drive_mode: np.ndarray  # (T,) str per element
    geometry: ParticleGeometry  # reference = union mask; bbox in full-frame
    metadata: Dict[str, Any] = field(default_factory=dict)
    crop_origin_xy: Tuple[int, int] = (0, 0)  # (x0, y0) full-frame origin of crop
    # crop shape (H, W) = soc_movie_tyx.shape[1:]; full-frame (x, y) = (crop_origin_xy[0] + x_local, crop_origin_xy[1] + y_local)


@dataclass
class RateMaps:
    """
    dc/dt and validity. Time dimension has length T-1 (one fewer than SOC frames).
    dc_dt_tyx[i] is the rate between frame i and frame i+1, aligned to the midpoint
    time_mid_s[i] = (time_s[i] + time_s[i+1]) / 2. All plotting and downstream use
    must treat dc/dt as T-1 and time_mid_s as the corresponding time axis.
    """

    dc_dt_tyx: np.ndarray  # (T-1, Y, X) rate at midpoints; length T-1
    time_mid_s: np.ndarray  # (T-1,) midpoint times; length T-1
    valid_mask_tyx: np.ndarray  # (T-1, Y, X) bool
    dt_s: np.ndarray  # (T-1,) Δt for each interval
    smoothing_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DescriptiveSummary:
    """Aggregated descriptive metrics and output paths."""

    per_particle: List[Dict[str, Any]]  # one dict per particle; includes onset_soc, onset_rate, valid coverage
    global_metrics: Dict[str, Any]
    figure_paths: List[str] = field(default_factory=list)
    table_paths: List[str] = field(default_factory=list)
