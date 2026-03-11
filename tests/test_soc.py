"""Tests for SOC computation: denominator zero -> NaN, no zeros."""

import numpy as np
import pytest

from reaction_kinetics.soc import soc_from_heights, build_soc_movie


def test_soc_denominator_zero_is_nan():
    # Only where a1g_c + a1g_d == 0 do we get NaN
    a1g_c = np.array([1.0, 0.0, 2.0])
    a1g_d = np.array([1.0, 0.0, 0.0])  # index 1: 0+0=0 -> NaN; index 2: 2+0=2 -> 1.0
    out = soc_from_heights(a1g_c, a1g_d)
    assert np.isnan(out[1])
    assert out[0] == 0.5
    assert out[2] == 1.0


def test_soc_no_silent_zero():
    a1g_c = np.array([0.0, 0.0])
    a1g_d = np.array([0.0, 0.0])
    out = soc_from_heights(a1g_c, a1g_d)
    assert np.all(np.isnan(out))
    assert not np.any(out == 0)


def test_build_soc_movie():
    c = np.ones((2, 2, 2))
    d = np.ones((2, 2, 2))
    c[0, 0, 0] = 0
    d[0, 0, 0] = 0
    soc, valid = build_soc_movie(c, d)
    assert soc.shape == (2, 2, 2)
    assert np.isnan(soc[0, 0, 0])
    assert valid[0, 0, 0] == False
    assert np.isfinite(soc[0, 0, 1]).all() or np.isfinite(soc[0, 1, 0]).all()
