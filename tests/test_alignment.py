"""Tests for ElectrochemTrace and drive inference from voltage."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from reaction_kinetics.alignment import infer_voltage_trend, load_timebounds, build_electrochem_trace


def test_infer_voltage_trend():
    # Rising then falling
    v = np.concatenate([np.linspace(3, 4, 5), np.linspace(4, 3, 5)])
    trend, sign = infer_voltage_trend(v)
    assert np.any(trend == "increasing")
    assert np.any(trend == "decreasing")
    assert sign[0] == 1
    assert sign[-1] == -1


def test_build_electrochem_trace_requires_file():
    with pytest.raises(FileNotFoundError):
        build_electrochem_trace(Path("/nonexistent/path"))


def test_load_timebounds_integration():
    # Use repo data if present
    data_root = Path("data")
    if (data_root / "voltage_profiles" / "voltage_profile_timestep_avg_voltage_with_timebounds.csv").exists():
        df = load_timebounds(data_root)
        assert "timestep" in df.columns
        assert "avg_Ewe/V" in df.columns
        assert len(df) >= 1
