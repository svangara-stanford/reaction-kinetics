"""Geometry from (Y,X) mask: area, centroid, bbox, boundary, distance-to-boundary."""

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from reaction_kinetics.config import DX_UM, DY_UM
from reaction_kinetics.schema import ParticleGeometry


def bbox_from_mask(mask_xy: np.ndarray) -> Tuple[int, int, int, int]:
    """(x_min, y_min, x_max, y_max) inclusive from (Y,X) bool mask."""
    rows = np.any(mask_xy, axis=1)
    cols = np.any(mask_xy, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 0, 0, 0, 0
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(x_min), int(y_min), int(x_max), int(y_max)


def centroid_from_mask(mask_xy: np.ndarray) -> Tuple[float, float]:
    """(x_center, y_center) in pixel coordinates. (Y,X) mask."""
    ys, xs = np.where(mask_xy)
    if len(xs) == 0:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.mean(ys))


def boundary_mask(mask_xy: np.ndarray) -> np.ndarray:
    """Boundary pixels = mask pixels with at least one non-mask neighbor. (Y,X)."""
    structure = np.ones((3, 3), dtype=int)
    eroded = ndimage.binary_erosion(mask_xy, structure=structure)
    return mask_xy & ~eroded


def distance_to_boundary_px(mask_xy: np.ndarray) -> np.ndarray:
    """
    For each pixel in mask, distance in pixels to nearest boundary pixel.
    Boundary pixels get 0. Outside mask: NaN. (Y,X) float.
    """
    bound = boundary_mask(mask_xy)
    ny, nx = mask_xy.shape
    dist = np.full((ny, nx), np.nan, dtype=float)
    if not np.any(mask_xy):
        return dist
    # 1 inside mask, 0 at boundary; distance_transform_edt gives distance to nearest 0
    full_im = np.zeros((ny, nx), dtype=float)
    full_im[mask_xy] = 1
    full_im[bound] = 0
    d = ndimage.distance_transform_edt(full_im)
    dist[mask_xy] = d[mask_xy]
    return dist


def interior_mask(
    mask_xy: np.ndarray, erosion_px: int = 1
) -> Optional[np.ndarray]:
    """Erode mask by erosion_px; return (Y,X) bool or None if erosion_px<=0."""
    if erosion_px <= 0:
        return None
    structure = np.ones((2 * erosion_px + 1, 2 * erosion_px + 1), dtype=bool)
    return ndimage.binary_erosion(mask_xy, structure=structure)


def build_particle_geometry(
    particle_id: int,
    mask_union_xy: np.ndarray,
    mask_intersection_xy: Optional[np.ndarray] = None,
    dx_um: float = DX_UM,
    dy_um: float = DY_UM,
    erosion_for_interior_px: int = 0,
) -> ParticleGeometry:
    """
    Build ParticleGeometry from union mask (reference). All arrays (Y,X).
    """
    ny, nx = mask_union_xy.shape
    area_px = int(np.sum(mask_union_xy))
    area_um2 = area_px * dx_um * dy_um
    cx, cy = centroid_from_mask(mask_union_xy)
    centroid_xy_um = (cx * dx_um, cy * dy_um)
    bbox = bbox_from_mask(mask_union_xy)
    boundary_xy = boundary_mask(mask_union_xy)
    dist_px = distance_to_boundary_px(mask_union_xy)
    dist_um = np.full_like(dist_px, np.nan)
    np.copyto(dist_um, dist_px * dx_um, where=np.isfinite(dist_px))
    interior_xy = interior_mask(mask_union_xy, erosion_for_interior_px)

    return ParticleGeometry(
        particle_id=particle_id,
        mask_xy=mask_union_xy,
        mask_union_xy=mask_union_xy,
        mask_intersection_xy=mask_intersection_xy,
        boundary_xy=boundary_xy,
        area_px=area_px,
        area_um2=area_um2,
        centroid_xy_px=(cx, cy),
        centroid_xy_um=centroid_xy_um,
        bbox=bbox,
        distance_to_boundary_px=dist_px,
        distance_to_boundary_um=dist_um,
        interior_xy=interior_xy,
        shape_yx=(ny, nx),
    )
