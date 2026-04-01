# reaction-kinetics

Level 1 descriptive reaction-kinetics analysis from Raman-derived SOC movies of an NMC111 cathode. **Level 1** means quantifying which particles react earlier or later, which regions within a particle are fast or slow, how heterogeneity evolves, and whether behavior is synchronized—without yet fitting PDE-based or inverse models.

## SOC definition

SOC = `a1g_c height / (a1g_c height + a1g_d height)`. Where the denominator is zero, SOC is set to NaN and not replaced by zero.

## Added lithium-stoichiometry and Raman-weighted current allocation (proxies)

The Raman map covers only a **subregion** of the cathode. The measured applied current `I_tot` is a **global** cell value. Current-allocation outputs in this repo are therefore **Raman-weighted, scan-region-normalized relative current maps and proxies within the imaged grid**. They show how inferred local activity is **distributed across the scan** for visualization and comparison. They are **not** absolute local electrochemical current densities for the full cathode, and the A/cm² fields are **normalization-based density proxies**, not physically defensible whole-electrode local `j`.

- Charged-state proxy: `s(x,y,t) = H_c / (H_c + H_d)` (same numerical definition as SOC proxy)
- Lithium stoichiometry map: `x_li = (1-s)*1.13 + s*0.268 = 1.13 - 0.862*s`
- Stoichiometric rate map: `dx_li_dt` computed on midpoint axis with length `T-1`
- Scan-region relative weights (default): `w_i(t) = |dx_i/dt| / sum_k |dx_k/dt|`
- Scan-region allocated current (normalization): `I_i = I_tot * w_i`, with `I_tot = 219.164 uA` (allocation for plotting, not “all current flows only through these pixels”)
- Scan-region current-density **proxy**: `j_proxy_i = I_i / A_pix`, `A_pix = 2.5e-9 cm^2`

**Per-particle current shares** sum the same global weights `w` only on pixels **exclusively** assigned to each particle (where union masks overlap, the **lowest particle_id** wins). Sums over retained particles are **≤ 1**; values **&lt; 1** indicate Raman activity outside those particles or outside masks. See `mask_overlap_audit.csv`, `current_share_audit.csv`, and diagnostic figures.

## Data limitations

- **No current magnitude:** The repository does not contain an explicit current column. Voltage-trend segments and turning points are inferred from `avg_Ewe/V` vs timestep. The pipeline uses `voltage_trend` and `drive_sign_proxy`; charge/discharge labels are only applied when the convention is explicitly documented (see `reaction_kinetics/CONVENTIONS.md`).
- Particle masks are discovered from filenames (`*_pixels.csv`); particle IDs on disk are 1–11 (no particle 0).
- Masks can vary by timestep: **per-timestep masks** are used for all framewise calculations; **reference geometry** is the **union** of masks across timesteps; the **intersection** mask is computed for robustness QA.

## Repo structure

- `reaction_kinetics/` — main package
- `scripts/run_descriptive_kinetics.py` — run the descriptive pipeline
- `data/` — counts, masks, voltage profiles (see plan for layout)
- `outputs/` — generated figures and tables (not committed)

## How to run

```bash
python scripts/run_descriptive_kinetics.py
```

Optional: `--data-root`, `--output-root`, `--smooth-window`, `--erode-boundary-px`, `--analysis-field {soc,x_li}`.

## Outputs

- **outputs/figures/** — dc/dt vs c, normalized rate maps, heterogeneity, particle SOC curves, boundary vs interior, onset/lag table, synchronization heatmap.
- **outputs/tables/** — particle_summary.csv, global_summary.csv, episode_metadata.csv, pixel_observations (full-frame x, y). Valid-data coverage (fraction valid SOC, fraction valid dc/dt per particle over time) and two onset metrics (SOC-threshold-based, mean-|dc/dt|-threshold-based) are included.
- **outputs/intermediate/** — Full-field SOC movie (`soc_full_field_tyx.npy`) for debugging; mask QA (union, intersection, occupancy frequency) per particle.
- Additional intermediates include `s_movie_full_field_tyx_T.npy`, `x_li_movie_full_field_tyx_T.npy`, `dx_li_dt_full_field_tyx_Tminus1.npy`, `relative_current_weight_full_field_tyx_Tminus1.npy`, `scan_region_allocated_current_a_full_field_tyx_Tminus1.npy`, and `scan_region_normalized_current_density_proxy_a_per_cm2_full_field_tyx_Tminus1.npy`.
- Tables: `mask_overlap_audit.csv`, `current_share_audit.csv`, `current_share_support_sensitivity.csv`, `scan_region_current_partition_timestep_summary.csv`, `per_particle_current_share_over_time_Tminus1.csv`, `per_particle_current_share_summary.csv`; partition flags include `partition_invalid_frame` and `near_zero_denominator_frame` where relevant. `pixel_observations` includes `scan_region_allocated_current_a`, `scan_region_normalized_current_density_proxy_a_per_cm2`, and T−1 partition flags.
- If you have older intermediate `.npy` files from before the allocated/proxy rename, replace them by re-running the pipeline; see `CURRENT_SHARE_FIX_NOTES.md` for semantics.
- **Deprecated (do not use)** intermediate names from older runs: `assigned_local_current_a_full_field_tyx_Tminus1.npy`, `scan_region_normalized_current_density_a_per_cm2_full_field_tyx_Tminus1.npy`. Current outputs use `scan_region_allocated_current_a_full_field_tyx_Tminus1.npy` and `scan_region_normalized_current_density_proxy_a_per_cm2_full_field_tyx_Tminus1.npy`.
- Support audit: `support_region_breakdown.csv`, `weight_region_breakdown.csv`, `area_vs_weight_comparison.csv`, `boundary_weight_summary.csv`, `support_interpretation_note.md`; figures `weight_region_breakdown_vs_time.png`, `area_vs_weight_comparison.png`, `boundary_weight_vs_interior.png`.

**dc/dt** has length T-1 and is aligned to time midpoints; see `reaction_kinetics/CONVENTIONS.md`.

## Next steps

Toward explicit kinetic modeling (e.g. PDE-constrained inversion) in a later version.
