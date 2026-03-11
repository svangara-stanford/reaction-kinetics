"""
Load timebounds and build ElectrochemTrace. Infer voltage_trend and drive_sign_proxy
from avg_Ewe/V vs timestep (turning points). Optionally apply charge/discharge convention
(documented in CONVENTIONS.md): rising V -> charge, falling V -> discharge.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from reaction_kinetics.config import TIMEBOUNDS_FILENAME, VOLTAGE_DIR
from reaction_kinetics.schema import ElectrochemTrace


# When True, set drive_mode ("charge"|"discharge") and drive_sign from voltage trend.
# Convention: rising voltage = charge (+1), falling = discharge (-1). See CONVENTIONS.md.
APPLY_CHARGE_DISCHARGE_CONVENTION: bool = True


def load_timebounds(data_root: Path) -> pd.DataFrame:
    """Load voltage_profile_timestep_avg_voltage_with_timebounds.csv."""
    path = data_root / VOLTAGE_DIR / TIMEBOUNDS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Timebounds file not found: {path}")
    df = pd.read_csv(path)
    for col in ["timestep", "start_time_h", "end_time_h", "avg_Ewe/V"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    return df.sort_values("timestep").reset_index(drop=True)


def infer_voltage_trend(voltage_v: np.ndarray) -> tuple:
    """
    Infer voltage_trend and drive_sign_proxy from voltage vs timestep.
    Returns (voltage_trend, drive_sign_proxy) both (T,) arrays.
    - voltage_trend: "increasing" | "decreasing" | "unknown"
    - drive_sign_proxy: +1 (V rising), -1 (V falling), 0 (unknown)
    """
    T = len(voltage_v)
    voltage_trend = np.full(T, "unknown", dtype=object)
    drive_sign_proxy = np.zeros(T, dtype=np.int64)

    if T < 2:
        return voltage_trend, drive_sign_proxy

    # Use forward difference: slope to next timestep
    diff = np.diff(voltage_v)
    # For last timestep use previous slope
    slope = np.zeros(T)
    slope[:-1] = diff
    slope[-1] = diff[-1] if len(diff) else 0

    inc = slope > 1e-9
    dec = slope < -1e-9
    voltage_trend[inc] = "increasing"
    voltage_trend[dec] = "decreasing"
    drive_sign_proxy[inc] = 1
    drive_sign_proxy[dec] = -1

    return voltage_trend, drive_sign_proxy


def build_electrochem_trace(
    data_root: Path,
    timestep_indices: Optional[List[int]] = None,
) -> ElectrochemTrace:
    """
    Build ElectrochemTrace from timebounds file. If timestep_indices is given,
    align to those timesteps (same length as SOC movie); else use all rows.
    """
    df = load_timebounds(data_root)
    if timestep_indices is not None:
        # Reindex to match SOC movie timesteps
        ts_set = set(df["timestep"].values)
        for t in timestep_indices:
            if t not in ts_set:
                raise ValueError(f"Timestep {t} in movie not found in timebounds")
        df = df.set_index("timestep").loc[timestep_indices].reset_index()
    df = df.sort_values("timestep").reset_index(drop=True)

    time_h = (df["start_time_h"].values + df["end_time_h"].values) / 2
    time_s = time_h * 3600.0
    voltage_v = df["avg_Ewe/V"].values
    timestep = df["timestep"].values
    timestep_start_h = df["start_time_h"].values
    timestep_end_h = df["end_time_h"].values

    voltage_trend, drive_sign_proxy = infer_voltage_trend(voltage_v)

    if APPLY_CHARGE_DISCHARGE_CONVENTION:
        drive_mode = np.where(
            voltage_trend == "increasing",
            "charge",
            np.where(voltage_trend == "decreasing", "discharge", "unknown"),
        )
        drive_sign = drive_sign_proxy.copy()
    else:
        drive_mode = np.full(len(voltage_v), "unknown", dtype=object)
        drive_sign = np.zeros(len(voltage_v), dtype=np.int64)

    return ElectrochemTrace(
        time_h=time_h,
        time_s=time_s,
        voltage_v=voltage_v,
        timestep=timestep,
        timestep_start_h=timestep_start_h,
        timestep_end_h=timestep_end_h,
        voltage_trend=voltage_trend,
        drive_sign_proxy=drive_sign_proxy,
        drive_mode=drive_mode,
        drive_sign=drive_sign,
        is_constant_current_assumed=False,
    )
