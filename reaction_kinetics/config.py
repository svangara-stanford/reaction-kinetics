"""Pipeline configuration: paths and processing defaults. No hardcoded absolute paths."""

from pathlib import Path
from typing import Optional


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

# Expected full-frame grid
GRID_NX: int = 30
GRID_NY: int = 30

# Rate estimation defaults
DEFAULT_SMOOTH_WINDOW: int = 3  # Savitzky-Golay window (odd)
DEFAULT_ERODE_BOUNDARY_PX: int = 0  # 0 = no erosion
DEFAULT_SPATIAL_SMOOTHING: bool = False

# Onset metrics (two definitions)
ONSET_SOC_DELTA_THRESHOLD: float = 0.05  # SOC-threshold-based: first t where mean SOC departs from initial by this
# Rate-based onset: threshold = max(floor, fraction_of_max * max mean |dc/dt|) so it works when dc/dt ~ 1e-3
ONSET_MEAN_ABSRATE_THRESHOLD: float = 1e-6  # floor to avoid noise-triggered onset
ONSET_MEAN_ABSRATE_FRACTION_OF_MAX: float = 0.1  # first t where mean |dc/dt| >= this fraction of particle's max

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
