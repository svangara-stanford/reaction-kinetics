"""Exclusive per-particle aggregation of scan-region relative weights (no double counting)."""

from typing import Dict, List, Tuple

import numpy as np


def exclusive_pixel_owner_from_masks(
    particle_ids: List[int],
    mask_union_by_id: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    Assign each pixel to at most one particle: lowest particle_id wins where masks overlap.
    Unassigned pixels (no mask) get -1.
    """
    if not particle_ids:
        raise ValueError("particle_ids must be non-empty")
    first = particle_ids[0]
    ny, nx = mask_union_by_id[first].shape
    owner = np.full((ny, nx), -1, dtype=np.int32)
    for pid in sorted(particle_ids):
        m = mask_union_by_id[pid].astype(bool)
        claim = m & (owner < 0)
        owner[claim] = int(pid)
    return owner


def pairwise_mask_overlap_counts(
    particle_ids: List[int],
    mask_union_by_id: Dict[int, np.ndarray],
) -> List[Dict[str, int]]:
    """Raw pairwise overlap (intersection) before exclusive assignment; for audit only."""
    rows: List[Dict[str, int]] = []
    for i, pi in enumerate(particle_ids):
        for pj in particle_ids[i + 1 :]:
            mi = mask_union_by_id[pi].astype(bool)
            mj = mask_union_by_id[pj].astype(bool)
            n = int(np.sum(mi & mj))
            rows.append({
                "particle_i": int(pi),
                "particle_j": int(pj),
                "overlap_pixel_count": n,
            })
    return rows


def per_particle_share_timeseries(
    relative_weight_tyx: np.ndarray,
    pixel_owner: np.ndarray,
    particle_ids: List[int],
    partition_invalid_t: np.ndarray,
) -> Dict[int, np.ndarray]:
    """
    Sum relative weights over full field where pixel_owner == pid.

    Uses np.nansum on owned pixels so one NaN weight does not zero out the particle total.
    Share is NaN only when ``partition_invalid_t[t]``. No owned pixels => 0.0.
    """
    Tm1 = relative_weight_tyx.shape[0]
    out: Dict[int, np.ndarray] = {}
    for pid in particle_ids:
        sel = pixel_owner == int(pid)
        share = np.full(Tm1, np.nan, dtype=float)
        for t in range(Tm1):
            if partition_invalid_t[t]:
                continue
            if not np.any(sel):
                share[t] = 0.0
            else:
                share[t] = float(np.nansum(relative_weight_tyx[t][sel]))
        out[int(pid)] = share
    return out


def mean_weight_and_j_over_owned_pixels(
    w_tyx: np.ndarray,
    j_tyx: np.ndarray,
    pixel_owner: np.ndarray,
    pid: int,
    partition_invalid_t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-timestep spatial mean of w and j on pixels exclusively owned by pid."""
    Tm1 = w_tyx.shape[0]
    sel = pixel_owner == int(pid)
    mean_w = np.full(Tm1, np.nan, dtype=float)
    mean_j = np.full(Tm1, np.nan, dtype=float)
    if not np.any(sel):
        return mean_w, mean_j
    for t in range(Tm1):
        if partition_invalid_t[t]:
            continue
        mean_w[t] = float(np.nanmean(w_tyx[t][sel]))
        mean_j[t] = float(np.nanmean(j_tyx[t][sel]))
    return mean_w, mean_j


def count_owned_pixels(pixel_owner: np.ndarray, particle_ids: List[int]) -> int:
    """Pixels assigned to any particle in this run."""
    s = set(particle_ids)
    return int(np.sum(np.isin(pixel_owner, list(s))))
