"""Tests for particle mask parsing and mask-to-array conversion."""

import numpy as np
import pytest

from reaction_kinetics.masks import parse_pixel_string, pixels_to_mask


def test_parse_pixel_string():
    s = "25,3;26,3;24,4;25,4"
    out = parse_pixel_string(s)
    assert out == {(25, 3), (26, 3), (24, 4), (25, 4)}


def test_parse_pixel_string_empty():
    assert parse_pixel_string("") == set()
    assert parse_pixel_string("   ") == set()


def test_parse_pixel_string_single():
    assert parse_pixel_string("0,0") == {(0, 0)}


def test_pixels_to_mask():
    pixels = {(1, 1), (2, 1), (1, 2)}
    mask = pixels_to_mask(pixels, ny=5, nx=5)
    assert mask.shape == (5, 5)
    assert np.sum(mask) == 3
    assert mask[1, 1] and mask[2, 1] and mask[1, 2]


def test_pixels_to_mask_out_of_bounds_ignored():
    pixels = {(0, 0), (100, 100)}
    mask = pixels_to_mask(pixels, ny=30, nx=30)
    assert mask[0, 0]
    assert np.sum(mask) == 1
