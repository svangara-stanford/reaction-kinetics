"""Tests for dc/dt computation: shape T-1, valid mask, NaN propagation."""

import numpy as np
import pytest

from reaction_kinetics.rate import compute_dc_dt


def test_rate_shape_t_minus_1():
    T, ny, nx = 5, 3, 3
    soc = np.random.rand(T, ny, nx).astype(float)
    time_s = np.arange(T, dtype=float) * 10.0
    rate = compute_dc_dt(soc, time_s)
    assert rate.dc_dt_tyx.shape[0] == T - 1
    assert rate.time_mid_s.shape[0] == T - 1
    assert rate.dt_s.shape[0] == T - 1


def test_rate_nan_propagation():
    T, ny, nx = 4, 2, 2
    soc = np.ones((T, ny, nx))
    soc[1, 0, 0] = np.nan
    time_s = np.arange(T, dtype=float)
    rate = compute_dc_dt(soc, time_s)
    # Interval 0->1: one pixel has NaN at t=1, so dc/dt there should be invalid/NaN
    assert np.any(np.isnan(rate.dc_dt_tyx)) or not rate.valid_mask_tyx[0, 0, 0]


def test_rate_valid_mask():
    soc = np.ones((3, 2, 2))
    time_s = np.array([0.0, 1.0, 2.0])
    rate = compute_dc_dt(soc, time_s)
    assert rate.valid_mask_tyx.shape == (2, 2, 2)
    assert np.all(rate.valid_mask_tyx)  # all finite
