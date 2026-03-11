# reaction-kinetics

Level 1 descriptive reaction-kinetics analysis from Raman-derived SOC movies of an NMC111 cathode. **Level 1** means quantifying which particles react earlier or later, which regions within a particle are fast or slow, how heterogeneity evolves, and whether behavior is synchronized—without yet fitting PDE-based or inverse models.

## SOC definition

SOC = `a1g_c height / (a1g_c height + a1g_d height)`. Where the denominator is zero, SOC is set to NaN and not replaced by zero.

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

Optional: `--data-root`, `--output-root`, `--smooth-window`, `--erode-boundary-px`.

## Outputs

- **outputs/figures/** — dc/dt vs c, normalized rate maps, heterogeneity, particle SOC curves, boundary vs interior, onset/lag table, synchronization heatmap.
- **outputs/tables/** — particle_summary.csv, global_summary.csv, episode_metadata.csv, pixel_observations (full-frame x, y). Valid-data coverage (fraction valid SOC, fraction valid dc/dt per particle over time) and two onset metrics (SOC-threshold-based, mean-|dc/dt|-threshold-based) are included.
- **outputs/intermediate/** — Full-field SOC movie (`soc_full_field_tyx.npy`) for debugging; mask QA (union, intersection, occupancy frequency) per particle.

**dc/dt** has length T-1 and is aligned to time midpoints; see `reaction_kinetics/CONVENTIONS.md`.

## Next steps

Toward explicit kinetic modeling (e.g. PDE-constrained inversion) in a later version.
