"""Pipeline configuration: paths and processing defaults. No hardcoded absolute paths."""

from pathlib import Path
from typing import List, Optional


def get_data_root(override: Optional[str] = None) -> Path:
    """Data root: override if provided, else default relative to repo (script cwd)."""
    if override is not None:
        return Path(override).resolve()
    return Path("data").resolve()


def get_output_root(override: Optional[str] = None) -> Path:
    """Output root for figures and tables."""
    if override is not None:
        return Path(override).resolve()
    return Path("outputs").resolve()


# Spatial scale (microns per pixel)
DX_UM: float = 0.5
DY_UM: float = 0.5
PIXEL_AREA_CM2: float = (DX_UM * 1e-4) * (DY_UM * 1e-4)

# Physically grounded stoichiometry/current parameters
I_TOT_A: float = 219.164e-6
LI_STOICH_PRISTINE: float = 1.13
LI_STOICH_CHARGED: float = 0.268

# If sum(|dx_li_dt|) over the partition support is positive but below this absolute
# threshold, treat the frame as invalid (NaN weights) to avoid normalizing noise.
SCAN_REGION_SUM_ABS_DX_LI_DT_EPS: float = 1e-14

# Expected full-frame grid
GRID_NX: int = 30
GRID_NY: int = 30

# Rate estimation defaults
DEFAULT_SMOOTH_WINDOW: int = 3  # Savitzky-Golay window (odd)
DEFAULT_ERODE_BOUNDARY_PX: int = 0  # 0 = no erosion
DEFAULT_SPATIAL_SMOOTHING: bool = False

# Onset metrics (two definitions)
ONSET_SOC_DELTA_THRESHOLD: float = 0.05  # SOC-threshold-based: first t where mean SOC departs from initial by this
# Rate-based onset: particle baseline + smoothed rate + delta above baseline
ONSET_RATE_BASELINE_N_FIRST: int = 10  # baseline = median/mean of first N mean_abs_rate_t
ONSET_RATE_BASELINE_METHOD: str = "median"  # "median" or "mean"
ONSET_RATE_SMOOTH_WINDOW: int = 5  # odd; Savitzky-Golay or rolling mean
ONSET_RATE_DELTA_FRACTION: float = 0.2  # thresh = baseline + this * (max_smooth - baseline)

# Persistent normalized rate: time-window correlation
N_RATE_PERSISTENCE_WINDOWS: int = 4
# Use one global symmetric color scale across all particles in the grid plot
PERSISTENT_NORM_RATE_USE_GLOBAL_SCALE: bool = False

# Boundary vs interior: interior = distance_to_boundary_px >= this (px)
INTERIOR_THRESHOLD_PX: float = 1.0

# Particle filter: None = use all discovered; [1..8] = restrict to particles 1-8 (default for this project)
PARTICLE_IDS_INCLUDE: Optional[List[int]] = [1, 2, 3, 4, 5, 6, 7, 8]

# Boundary-only kinetics: exclude these particle IDs (retained = 2, 4, 5, 6, 7)
BOUNDARY_ANALYSIS_EXCLUDE_IDS: List[int] = [1, 3, 8]
# Common SOC grid size for boundary/full-particle fits
BOUNDARY_FIT_SOC_GRID_SIZE: int = 51

# Full-field SOC artifact (saved under output_root/intermediate for debugging)
FULL_FIELD_SOC_FILENAME: str = "soc_full_field_tyx.npy"

# Subdirs under data_root
COUNTS_A1G_DIR: str = "counts/counts_A1g"
COUNTS_CARBON_DIR: str = "counts/counts_carbon"
PARTICLE_MASKS_DIR: str = "particle_masks"
VOLTAGE_DIR: str = "voltage_profiles"
TIMEBOUNDS_FILENAME: str = "voltage_profile_timestep_avg_voltage_with_timebounds.csv"

# Subdirs under output_root
FIGURES_DIR: str = "figures"
TABLES_DIR: str = "tables"
INTERMEDIATE_DIR: str = "intermediate"
