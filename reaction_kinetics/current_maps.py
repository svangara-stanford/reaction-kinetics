"""Scan-region-normalized Raman-weighted current allocation (not full-cathode absolute j)."""

from typing import Dict, Optional

import numpy as np

from reaction_kinetics.config import SCAN_REGION_SUM_ABS_DX_LI_DT_EPS


def compute_scan_region_current_maps(
    dx_li_dt_tyx: np.ndarray,
    i_tot_a: float,
    pixel_area_cm2: float,
    support_mask_yx: Optional[np.ndarray] = None,
    denom_abs_eps: float = SCAN_REGION_SUM_ABS_DX_LI_DT_EPS,
) -> Dict[str, np.ndarray]:
    """
    Scan-region-normalized relative current allocation from |dx_li/dt| at each T-1 frame.

    If ``support_mask_yx`` (ny, nx) bool is provided, the denominator uses only pixels
    where the mask is True (tracked-particle-union support mode).

    Frames with non-finite, non-positive, or near-zero (below ``denom_abs_eps``)
    denominators are partition-invalid (NaN weights).
    """
    arr = dx_li_dt_tyx.astype(float)
    Tm1 = arr.shape[0]
    abs_arr = np.abs(arr)
    finite = np.isfinite(abs_arr)
    if support_mask_yx is not None:
        sm = support_mask_yx.astype(bool)
        if sm.shape != finite.shape[1:]:
            raise ValueError(
                f"support_mask_yx shape {sm.shape} != spatial shape {finite.shape[1:]}"
            )
        finite_use = finite & sm[np.newaxis, :, :]
    else:
        finite_use = finite

    denom_t = np.nansum(np.where(finite_use, abs_arr, np.nan), axis=(1, 2))
    zero_denominator_t = (~np.isfinite(denom_t)) | (denom_t <= 0)
    near_zero_denominator_t = (
        np.isfinite(denom_t) & (denom_t > 0) & (denom_t < float(denom_abs_eps))
    )
    partition_invalid_t = zero_denominator_t | near_zero_denominator_t

    w_tyx = np.full_like(arr, np.nan, dtype=float)
    for t in range(Tm1):
        if partition_invalid_t[t]:
            continue
        w_tyx[t] = np.where(
            finite_use[t],
            abs_arr[t] / denom_t[t],
            np.nan,
        )

    allocated_i_tyx = i_tot_a * w_tyx
    density_proxy_tyx = allocated_i_tyx / float(pixel_area_cm2)

    valid_count_t = np.sum(finite_use, axis=(1, 2)).astype(int)
    return {
        "relative_current_weight_tyx": w_tyx,
        "scan_region_allocated_current_a_tyx": allocated_i_tyx,
        "scan_region_normalized_current_density_proxy_a_per_cm2_tyx": density_proxy_tyx,
        "sum_abs_dx_li_dt_t": denom_t,
        "zero_denominator_t": zero_denominator_t.astype(bool),
        "near_zero_denominator_t": near_zero_denominator_t.astype(bool),
        "partition_invalid_t": partition_invalid_t.astype(bool),
        "valid_pixel_count_t": valid_count_t,
    }
