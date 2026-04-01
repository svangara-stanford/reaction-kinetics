"""SOC and lithium stoichiometry from A1g counts."""

from typing import Tuple

import numpy as np

from reaction_kinetics.utils import safe_divide
from reaction_kinetics.config import LI_STOICH_CHARGED, LI_STOICH_PRISTINE


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


def charged_fraction_proxy_from_heights(
    a1g_c_height: np.ndarray, a1g_d_height: np.ndarray
) -> np.ndarray:
    """
    Charged-state fraction proxy s(x,y,t) from A1g heights.
    Alias of SOC proxy for explicit physical naming.
    """
    return soc_from_heights(a1g_c_height, a1g_d_height)


def x_li_from_charged_fraction(
    s_tyx: np.ndarray,
    li_pristine: float = LI_STOICH_PRISTINE,
    li_charged: float = LI_STOICH_CHARGED,
) -> np.ndarray:
    """
    Lithium stoichiometry map:
        x_li = (1-s)*li_pristine + s*li_charged
    """
    s = s_tyx.astype(float)
    return (1.0 - s) * float(li_pristine) + s * float(li_charged)


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
