# Conventions

## Voltage trend and charge/discharge (no current)

- **Source:** Only `avg_Ewe/V` vs timestep is available; no current column.
- **Primary fields:** `voltage_trend` ("increasing" | "decreasing" | "unknown") and `drive_sign_proxy` (+1 when V rising, -1 when V falling, 0 unknown), inferred from turning points in voltage vs timestep.
- **Charge/discharge labels:** `drive_mode` ("charge" | "discharge" | "unknown") and `drive_sign` are only set when the following convention is explicitly applied and documented in code (e.g. in `alignment.py`):
  - **Convention:** Rising voltage segment → label "charge" and drive_sign +1; falling voltage segment → label "discharge" and drive_sign -1. This assumes the experiment is galvanostatic with positive current = charge (NMC111 cathode).
- If the convention is not applied, downstream code should use `voltage_trend` and `drive_sign_proxy` only.

## dc/dt time alignment

- **Length:** dc/dt has **T-1** frames (one fewer than SOC).
- **Alignment:** dc_dt_tyx[i] corresponds to the midpoint time `time_mid_s[i] = (time_s[i] + time_s[i+1]) / 2`.
- All plots and tables that use dc/dt must use the T-1 time axis (time_mid_s), not the T-length SOC time axis.

## x_li and current-map alignment

- `x_li_movie_tyx` has length **T** (same as SOC/s proxy).
- `dx_li_dt_tyx` has length **T-1** and is aligned to midpoint times.
- `relative_current_weight_tyx`, `scan_region_allocated_current_a_tyx`, and `scan_region_normalized_current_density_proxy_a_per_cm2_tyx` are all **T-1** quantities.
- Filenames and labels explicitly include `_T` or `_Tminus1` to avoid ambiguity.

## Scan-region-normalized current convention

- Primary partition uses magnitude: `|dx_li_dt|` (non-negative weights summing to 1 per frame over finite pixels).
- Normalization is scan-region-wide at each timestep across all finite scanned pixels.
- If `sum(|dx_li_dt|) == 0` for a frame, weight/allocation/proxy maps are set to NaN for that frame and flagged in summary tables.
- Allocated current and A/cm² fields are **scan-region normalization proxies** for visualization, not absolute full-cathode local current density.
- **Per-particle shares** use the same global weights summed only on pixels **exclusively** attributed to that particle: where union masks overlap, **lowest `particle_id` wins**, so shares do not double-count and their sum over particles is **≤ 1**.

## Coordinates

- **Full-frame:** (x, y) in the 30×30 scan; used in long-form pixel table and geometry.
- **Crop/local:** (x_local, y_local) within the particle bbox; used inside ParticleEpisode arrays. Conversion: `x = crop_origin_xy[0] + x_local`, `y = crop_origin_xy[1] + y_local`.

## Masks

- **Per-timestep masks:** Used for all framewise calculations (SOC validity, dc/dt validity).
- **Reference geometry mask:** Union of per-timestep masks over all timesteps; used for area, centroid, boundary, bbox, distance-to-boundary.
- **Intersection mask:** Optional; computed for robustness diagnostics and mask QA outputs.

## Carbon data

- Carbon count data is optional in v1; no carbon-kinetics correlation analysis.
