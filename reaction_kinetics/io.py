"""Load A1g count CSVs by timestep; validate grid shape."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from reaction_kinetics.config import GRID_NX, GRID_NY, COUNTS_A1G_DIR
from reaction_kinetics.utils import check_grid_shape


def list_a1g_timestep_files(data_root: Path) -> List[Path]:
    """Discover counts_A1g CSV files; return paths sorted by integer timestep."""
    counts_dir = data_root / COUNTS_A1G_DIR
    if not counts_dir.is_dir():
        raise FileNotFoundError(f"Counts directory not found: {counts_dir}")
    files: List[Tuple[int, Path]] = []
    for p in counts_dir.iterdir():
        if p.suffix.lower() != ".csv":
            continue
        stem = p.stem
        try:
            ts = int(stem)
        except ValueError:
            continue
        files.append((ts, p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def load_a1g_frame(path: Path) -> pd.DataFrame:
    """Load one timestep CSV. Validates required columns."""
    df = pd.read_csv(path)
    for col in ["x", "y", "a1g_c height", "a1g_d height"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    return df


def frame_to_grid(
    df: pd.DataFrame, expected_nx: int = GRID_NX, expected_ny: int = GRID_NY
) -> Tuple[int, int]:
    """Validate grid and return (ny, nx). Pivot: row order is (0,0)..(nx-1,0), (0,1).. so y is slowest."""
    x = df["x"].values
    y = df["y"].values
    ny, nx = check_grid_shape(x, y, expected_nx, expected_ny)
    return ny, nx


def load_a1g_movie(
    data_root: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Load all A1g frames and return SOC movie (T,Y,X), a1g_c height (T,Y,X), a1g_d height (T,Y,X),
    and list of timestep indices. SOC is not computed here (see soc.py).
    """
    files = list_a1g_timestep_files(data_root)
    if not files:
        raise FileNotFoundError(f"No A1g CSV files in {data_root / COUNTS_A1G_DIR}")

    timesteps: List[int] = []
    frames_soc: List[np.ndarray] = []
    frames_c: List[np.ndarray] = []
    frames_d: List[np.ndarray] = []
    ny, nx = None, None

    for p in files:
        ts = int(p.stem)
        timesteps.append(ts)
        df = load_a1g_frame(p)
        nyr, nxr = frame_to_grid(df, GRID_NX, GRID_NY)
        if ny is None:
            ny, nx = nyr, nxr
        elif (nyr, nxr) != (ny, nx):
            raise ValueError(f"Grid shape mismatch at {p}: expected ({ny},{nx}), got ({nyr},{nxr})")

        # Row order: x varies fastest, then y. So row i = (x_i, y_i) with x_i = i % nx, y_i = i // nx
        # DataFrame rows are typically in (x,y) order: (0,0),(1,0),...(nx-1,0),(0,1),...
        x = df["x"].values
        y = df["y"].values
        # Build (Y,X) so that arr[y, x] = value at (x,y)
        c = np.full((ny, nx), np.nan)
        d = np.full((ny, nx), np.nan)
        for i in range(len(df)):
            xi, yi = int(x[i]), int(y[i])
            c[yi, xi] = df["a1g_c height"].iloc[i]
            d[yi, xi] = df["a1g_d height"].iloc[i]
        frames_c.append(c)
        frames_d.append(d)

    T = len(frames_c)
    a1g_c_tyx = np.stack(frames_c, axis=0)
    a1g_d_tyx = np.stack(frames_d, axis=0)
    return a1g_c_tyx, a1g_d_tyx, timesteps, (ny, nx)
