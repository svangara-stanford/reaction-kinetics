"""Particle mask parsing: per-timestep masks, union (reference), intersection for QA."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from reaction_kinetics.config import GRID_NX, GRID_NY, PARTICLE_MASKS_DIR


def parse_pixel_string(pixels: str) -> Set[Tuple[int, int]]:
    """
    Parse semicolon-separated 'x,y;x,y;...' into set of (x, y).
    Handles empty string and missing/empty rows (returns empty set).
    """
    out: Set[Tuple[int, int]] = set()
    if not isinstance(pixels, str) or not pixels.strip():
        return out
    for part in pixels.strip().split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            xs, ys = part.split(",", 1)
            x, y = int(xs.strip()), int(ys.strip())
            out.add((x, y))
        except (ValueError, AttributeError):
            continue
    return out


def pixels_to_mask(
    pixels: Set[Tuple[int, int]], ny: int = GRID_NY, nx: int = GRID_NX
) -> np.ndarray:
    """Build (Y,X) boolean mask from set of (x, y). Full-frame coordinates."""
    mask = np.zeros((ny, nx), dtype=bool)
    for (x, y) in pixels:
        if 0 <= y < ny and 0 <= x < nx:
            mask[y, x] = True
    return mask


def discover_particle_ids(data_root: Path) -> List[int]:
    """Discover particle IDs from *_pixels.csv filenames. Does not assume particle 0 exists."""
    masks_dir = data_root / PARTICLE_MASKS_DIR
    if not masks_dir.is_dir():
        return []
    ids: List[int] = []
    for p in masks_dir.iterdir():
        if p.suffix.lower() != ".csv":
            continue
        stem = p.stem  # e.g. "1_pixels" -> need "1"
        if "_pixels" in stem:
            try:
                pid = int(stem.replace("_pixels", ""))
                ids.append(pid)
            except ValueError:
                continue
        elif stem.isdigit():
            ids.append(int(stem))
    return sorted(set(ids))


def load_particle_masks(
    data_root: Path,
    particle_id: int,
    ny: int = GRID_NY,
    nx: int = GRID_NX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Load per-timestep masks for one particle. Returns:
    - masks_tyx: (T, Y, X) bool, per-timestep mask
    - mask_union_xy: (Y, X) bool, union across all timesteps (reference geometry)
    - mask_intersection_xy: (Y, X) bool, intersection (for QA)
    - occupancy_xy: (Y, X) float in [0,1], fraction of timesteps where pixel is in mask
    - metadata: dict with e.g. masks_constant (bool), num_timesteps
    """
    masks_dir = data_root / PARTICLE_MASKS_DIR
    # Support both "1_pixels.csv" and "1.csv" style
    for name in (f"{particle_id}_pixels.csv", f"{particle_id}.csv"):
        path = masks_dir / name
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"No mask file for particle {particle_id} in {masks_dir}")

    df = pd.read_csv(path)
    if "timestep" not in df.columns or "pixels" not in df.columns:
        raise ValueError(f"Mask file must have 'timestep' and 'pixels' columns: {path}")

    df = df.sort_values("timestep").reset_index(drop=True)
    T = len(df)
    list_of_masks: List[np.ndarray] = []
    for _, row in df.iterrows():
        pixels = parse_pixel_string(row["pixels"])
        mask = pixels_to_mask(pixels, ny, nx)
        list_of_masks.append(mask)

    masks_tyx = np.stack(list_of_masks, axis=0)
    mask_union_xy = np.any(masks_tyx, axis=0)
    mask_intersection_xy = np.all(masks_tyx, axis=0)
    occupancy_xy = masks_tyx.astype(np.float64).sum(axis=0) / max(1, T)

    masks_constant = np.all(masks_tyx == masks_tyx[0:1], axis=0).all()
    metadata = {
        "masks_constant": bool(masks_constant),
        "num_timesteps": T,
        "timesteps": df["timestep"].values.tolist(),
    }

    return masks_tyx, mask_union_xy, mask_intersection_xy, occupancy_xy, metadata


def load_all_particle_masks(
    data_root: Path,
    particle_ids: Optional[List[int]] = None,
    ny: int = GRID_NY,
    nx: int = GRID_NX,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]]:
    """
    Load masks for all particles. Keys = particle_id; value = same as load_particle_masks.
    If particle_ids is None, discover from disk.
    """
    if particle_ids is None:
        particle_ids = discover_particle_ids(data_root)
    out: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict]] = {}
    for pid in particle_ids:
        try:
            out[pid] = load_particle_masks(data_root, pid, ny, nx)
        except FileNotFoundError:
            continue
    return out
