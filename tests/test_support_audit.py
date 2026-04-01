"""Support-region audit: count identities and weight conservation on toy grids."""

import numpy as np

from reaction_kinetics.support_audit import (
    compute_area_vs_weight_comparison,
    compute_support_region_breakdown,
    compute_weight_region_breakdown,
)


def test_support_counts_sum_and_union_matches_nonretained():
    ny, nx = 3, 3
    owner = np.full((ny, nx), -1, dtype=np.int32)
    owner[0, 0] = 1
    owner[0, 1] = 2
    union = np.zeros((ny, nx), dtype=bool)
    union[0, 0] = union[0, 1] = True
    dx = np.ones((1, ny, nx), dtype=float)
    dx[0, 1, 1] = np.nan
    pinv = np.array([False])
    time_mid_s = np.array([0.0])
    df = compute_support_region_breakdown(dx, owner, [1, 2], union, pinv, time_mid_s)
    row = df.iloc[0]
    assert row["n_valid_total"] == row["n_retained_particle_owned"] + row["n_nonretained_valid"]
    assert row["n_valid_outside_particle_union"] == row["n_nonretained_valid"]


def test_weight_split_sums_to_one_when_valid():
    ny, nx = 2, 2
    owner = np.array([[-1, 1], [1, -1]], dtype=np.int32)
    dx = np.ones((1, ny, nx), dtype=float)
    w = np.array([[[0.25, 0.25], [0.25, 0.25]]], dtype=float)
    pinv = np.array([False])
    time_mid_s = np.array([1.0])
    maps = {
        "relative_current_weight_tyx": w,
        "partition_invalid_t": pinv,
    }
    wf = compute_weight_region_breakdown(
        w, dx, owner, [1], pinv, time_mid_s
    )
    row = wf.iloc[0]
    assert abs(row["weight_on_retained_particles"] + row["weight_on_nonretained_valid_pixels"] - 1.0) < 1e-9


def test_area_vs_weight_merges():
    union = np.zeros((2, 2), dtype=bool)
    union[0, 0] = True
    support_df = compute_support_region_breakdown(
        np.ones((1, 2, 2)),
        np.array([[1, -1], [-1, -1]], dtype=np.int32),
        [1],
        union,
        np.array([False]),
        np.array([0.0]),
    )
    weight_df = compute_weight_region_breakdown(
        np.full((1, 2, 2), 0.25),
        np.ones((1, 2, 2)),
        np.array([[1, -1], [-1, -1]], dtype=np.int32),
        [1],
        np.array([False]),
        np.array([0.0]),
    )
    area_df = compute_area_vs_weight_comparison(support_df, weight_df)
    assert len(area_df) == 1
    assert np.isfinite(area_df.iloc[0]["retained_weight_per_unit_area"])
