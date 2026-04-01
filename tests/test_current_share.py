"""Exclusive current-share aggregation: no double counting, sum <= 1."""

import numpy as np

from reaction_kinetics.current_share import (
    exclusive_pixel_owner_from_masks,
    pairwise_mask_overlap_counts,
    per_particle_share_timeseries,
)


def test_exclusive_owner_sum_shares_le_one():
    """Overlapping masks: lower id wins; disjoint partition recovers full weight sum."""
    m1 = np.zeros((2, 2), dtype=bool)
    m2 = np.zeros((2, 2), dtype=bool)
    m1[0, 0] = m1[0, 1] = True
    m2[1, 0] = m2[1, 1] = True
    masks = {1: m1, 2: m2}
    owner = exclusive_pixel_owner_from_masks([1, 2], masks)
    assert owner[0, 0] == 1 and owner[1, 1] == 2

    w = np.full((1, 2, 2), 0.25, dtype=float)
    zero = np.array([False])
    shares = per_particle_share_timeseries(w, owner, [1, 2], zero)
    assert abs(shares[1][0] + shares[2][0] - 1.0) < 1e-9

    m1o = np.zeros((2, 2), dtype=bool)
    m2o = np.zeros((2, 2), dtype=bool)
    m1o[1, 1] = m2o[1, 1] = True
    m1o[0, 0] = True
    owner_o = exclusive_pixel_owner_from_masks([1, 2], {1: m1o, 2: m2o})
    assert owner_o[1, 1] == 1


def test_nansum_owned_pixels_sum_still_le_one():
    """NaN weight on one owned pixel must not wipe the particle share; sum stays <= 1."""
    m1 = np.zeros((2, 2), dtype=bool)
    m2 = np.zeros((2, 2), dtype=bool)
    m1[0, 0] = m1[0, 1] = True
    m2[1, 0] = m2[1, 1] = True
    owner = exclusive_pixel_owner_from_masks([1, 2], {1: m1, 2: m2})
    w = np.array([[[0.5, np.nan], [0.25, 0.25]]], dtype=float)
    ok = np.array([False])
    shares = per_particle_share_timeseries(w, owner, [1, 2], ok)
    assert shares[1][0] == 0.5
    assert shares[2][0] == 0.5
    assert shares[1][0] + shares[2][0] <= 1.0 + 1e-9


def test_pairwise_overlap_reports_intersection():
    m1 = np.zeros((2, 2), dtype=bool)
    m2 = np.zeros((2, 2), dtype=bool)
    m1[0, 0] = m1[0, 1] = True
    m2[0, 1] = m2[1, 1] = True
    rows = pairwise_mask_overlap_counts([1, 2], {1: m1, 2: m2})
    assert len(rows) == 1
    assert rows[0]["overlap_pixel_count"] == 1
