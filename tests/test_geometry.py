"""Tests for boundary extraction and distance-to-boundary."""

import numpy as np
import pytest

from reaction_kinetics.geometry import (
    bbox_from_mask,
    centroid_from_mask,
    boundary_mask,
    distance_to_boundary_px,
    build_particle_geometry,
)


def test_bbox_from_mask():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    x0, y0, x1, y1 = bbox_from_mask(mask)
    assert (x0, y0, x1, y1) == (1, 1, 3, 3)


def test_centroid_from_mask():
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    cx, cy = centroid_from_mask(mask)
    assert (cx, cy) == (1.0, 1.0)


def test_boundary_mask():
    # 3x3 block: boundary = 8 pixels, interior = 1
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    bound = boundary_mask(mask)
    assert bound[2, 2] == False  # interior
    assert bound[1, 1] == True
    assert np.sum(bound) == 8


def test_distance_to_boundary():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    dist = distance_to_boundary_px(mask)
    assert np.nanmin(dist) == 0  # boundary has 0
    assert np.isnan(dist[0, 0])
    assert np.isfinite(dist[2, 2])  # center
