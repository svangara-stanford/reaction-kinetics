# Voltage–SOC proxy validation

This analysis tests whether the **Raman SOC proxy** shares temporal structure with the **electrochemical voltage** signal. The goal is **proxy validation**: to show shared temporal structure, reasonable lag, aligned turning points, and an interpretable voltage–SOC relationship — **not** to prove that voltage and SOC are identical.

For the NMC cathode in this dataset, the Raman SOC proxy represents cathode lithiation state, which is approximately **inversely related** to voltage. Physically:

- Voltage maxima correspond to SOC minima.
- Voltage minima correspond to SOC maxima.

Therefore, the refined validation focuses on **inverted voltage** (1 − normalized voltage) as the primary comparison signal for SOC.

## Primary validation signal

- **All-particle mean SOC (retained)** is the primary validation signal: the mean of `mean_soc_t` over retained particles only (particles 2, 4, 5, 6, 7; particles 1, 3, 8 are excluded, consistent with boundary analysis).

## Lag conventions

- Full-run lag (legacy): **Positive lag** = SOC lags voltage (the SOC peak occurs **later** than the voltage peak).
- Segment-wise inverted-voltage lag: **Positive lag** = SOC lags **inverted** voltage within a segment.
- Lags are reported in timesteps and in hours.

## Outputs

| Output | Description |
|--------|-------------|
| **Normalized overlays** | Per-particle and one global figure: normalized [0,1] SOC and voltage vs time (hours). `validation/overlay_particle_{id}.png`, `validation/overlay_global.png`. |
| **Inversion-aware overlays** | Global and per-particle overlays of SOC vs **inverted normalized voltage**. `validation/overlay_global_inverted_voltage.png`, `validation/overlay_per_particle_inverted_voltage.png`. |
| **Cross-correlation** | Peak correlation and best lag (timesteps, hours) for each retained particle and for the global (retained-mean) SOC. Optional plot: `validation/crosscorr_global.png`. Table: `validation_correlation_summary.csv`. |
| **Segment fits** | Time split into monotonic segments (rising/falling from `drive_sign`). Per-segment linear fits for voltage(t) and SOC(t); slopes and R². Table: `validation_segment_fits.csv`. |
| **Turning-point alignment** | Local maxima and minima in voltage and in retained-mean SOC; for each voltage event, nearest SOC event of same type and time offset (SOC time − voltage time). Table: `validation_turning_points.csv`. |
| **Frequency comparison** | FFT magnitude spectrum for voltage and retained-mean SOC; dominant period and (optionally) spectral centroid. Figure: `validation/spectrum_comparison.png`. |
| **Inversion-aware frequency comparison** | FFT magnitude spectrum for inverted voltage and retained-mean SOC; dominant period and centroid. Figure: `validation/spectrum_comparison_inverted_voltage.png`. |
| **Voltage vs SOC curves** | Per-particle and global: voltage (y) vs SOC (x), optionally colored by time. `validation/voltage_vs_soc_particle_{id}.png`, `validation/voltage_vs_soc_global.png`. |
| **Inverted-voltage vs SOC curves** | Global (and optionally per-particle): inverted normalized voltage vs SOC, typically monotonic if the proxy is good. `validation/inverted_voltage_vs_soc_global.png`. |
| **Legacy summary table** | `validation_summary.csv`: per particle and one "global" row with correlation_peak, best_lag_timestep, best_lag_hours, segment slopes/time constants, turning_point mean offset, dominant periods, and optional r2_linear_voltage_soc. |
| **Segment-wise inverted-voltage lag** | `segment_lag_summary.csv`: per particle and segment, segment-wise correlation and lag between SOC and inverted voltage. |
| **Segment-wise fit comparison** | `segment_fit_comparison.csv`: per particle and segment, linear fit slopes and R² for inverted voltage vs time and SOC vs time. |
| **Turning-point alignment (inversion-aware)** | `turning_point_alignment.csv`: per particle and event, offsets between voltage maxima/minima and SOC minima/maxima. |
| **Proxy validation summary** | `proxy_validation_summary.csv`: per particle and global summary of turning-point offsets, segment-wise lags, dominant periods, and short qualitative notes. |

## Dependencies

- `numpy`, `scipy` (for `argrelmax`/`argrelmin`), `matplotlib`. No additional packages.
