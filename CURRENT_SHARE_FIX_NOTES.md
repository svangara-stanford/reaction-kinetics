# Current-share aggregation: fixes and interpretation

## Partition validity

Per-frame relative weights are normalized by the scan-region denominator `sum(|dx_li_dt|)` over the active support (finite `dx_li_dt` pixels, optionally restricted by a support mask).

- **`zero_denominator_t`**: exact zero or non-finite sum (no usable partition).
- **`near_zero_denominator_t`**: sum below `SCAN_REGION_SUM_ABS_DX_LI_DT_EPS` in config (treated as invalid; avoids unstable weights).
- **`partition_invalid_t`**: logical OR of the above; shares and audit sums use **NaN** for that frame.

Timestep summary CSVs expose `near_zero_denominator_frame` and `partition_invalid_frame`; diagnostics and plots skip invalid frames using **`partition_invalid_t`**, not `zero_denominator_t` alone.

## Per-particle shares

`per_particle_share_timeseries` sums weights on exclusively owned pixels with **`np.nansum`**: a single NaN weight on an owned pixel no longer zeros the whole particle share. Frames with no owned pixels contribute **0.0**; invalid partition frames remain **NaN**.

Sums across particles use **`np.sum`** (not `nansum`) so missing particles are not silently dropped when auditing totals.

## Support sensitivity

`current_share_support_sensitivity.csv` repeats the audit for two denominators:

1. **`full_valid_scan`**: all finite-`dx_li_dt` pixels (same as main `scan_region_maps`).
2. **`tracked_particle_union_only`**: denominator restricted to the union of tracked particle masks.

Compare `sum_retained_particle_shares` and `uncovered_weight_fraction` between modes to see how much weight sits outside tracked regions.

## Intermediates

Full-field `.npy` names under `outputs/intermediate/` match the variables above (`allocated` / `normalized_current_density_proxy`). Older runs may use different filenames; re-run the pipeline for consistent artifacts.

**Deprecated filenames (not written by current code):** `assigned_local_current_a_full_field_tyx_Tminus1.npy`, `scan_region_normalized_current_density_a_per_cm2_full_field_tyx_Tminus1.npy`. Prefer the `scan_region_allocated_current_*` and `*_proxy_*` names in the README.
