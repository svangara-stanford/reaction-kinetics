"""SOC computation from A1g counts. NaN where denominator is zero."""

from typing import Tuple

import numpy as np

from reaction_kinetics.utils import safe_divide


def soc_from_heights(
    a1g_c_height: np.ndarray, a1g_d_height: np.ndarray
) -> np.ndarray:
    """
    SOC = a1g_c_height / (a1g_c_height + a1g_d_height).
    Returns NaN where denominator is zero; no silent zero substitution.
    """
    denom = a1g_c_height.astype(float) + a1g_d_height.astype(float)
    return safe_divide(
        a1g_c_height.astype(float), denom, out=np.empty_like(denom)
    )


def build_soc_movie(
    a1g_c_tyx: np.ndarray, a1g_d_tyx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build full-field SOC movie (T,Y,X) from a1g_c and a1g_d height movies.
    Returns (soc_tyx, valid_tyx) where valid_tyx is True where SOC is not NaN.
    """
    soc_tyx = soc_from_heights(a1g_c_tyx, a1g_d_tyx)
    valid_tyx = np.isfinite(soc_tyx)
    return soc_tyx, valid_tyx
