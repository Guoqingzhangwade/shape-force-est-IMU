# Progress Note — 2026-04-19

## What I did

### 1. Added `--mode formulation_compare` to `meas_model_compare.py`
- New mode runs all 4 EKF formulations on a fixed matched-model bank
  (same shapes + same noise seeds for every method — fully paired)
- Methods: Quat numeric, Quat analytic ref, SO(3) analytic, SO(3) numeric
- Fixed alpha = 1.0 (selected from previous alpha sweep study)
- New CLI args: `--formulation-alpha` (default 1.0), `--plot-formulation-compare`
- Prints a manuscript-ready comparison table with post-burn-in NIS as the
  primary consistency metric
- Saves `meas_model_comparison_results.csv` and `.json`
- P0 = diag([1.0, 0.5, 1.0, 0.5, 0.25]) as default (same as so3_scale_sweep)
- New helper functions: `run_formulation_compare()`, `print_formulation_table()`,
  `FORMULATION_LABELS`, `FORMULATION_ORDER`

### 2. Created `sensor_placement_study_final.py` (new script)
- Finalized matched-model sensor number / placement study
- Estimator: SO(3)-analytic EKF (same as selected in formulation comparison)
- Same P0, alpha, Q as the finalized comparison study
- 17 sensor configurations: 4 × 1-IMU, 6 × 2-IMU, 4 × 3-IMU, 2 × 4-IMU,
  1 × 5-IMU
- Fixed shape bank + fixed noise seeds shared across ALL configs (paired fairness)
- Outputs: summary table, best-per-count, CSV, JSON
- Optional plots: best-vs-count, convergence, all-curves, 2-IMU heatmap
- Backup script `sensor_placement_study.py` left completely untouched

### 3. Created `plot_sensor_placement_results.py` (new plotting-only script)
- Reads saved CSV/JSON — no simulation re-runs
- Generates manuscript-ready figures:
  A. Best RMSE vs sensor count (from `sensor_best_by_count.csv`)
  B. Sparse 2-IMU heatmap (from 6 predefined 2-IMU rows in main CSV)
  C. Horizontal bar chart (all 17 configurations)
- Fixed 2-IMU heatmap axis-centering: uses `pcolormesh` with cell-edge
  coordinates so tick labels land at cell centers, not boundaries
- Saves PNG + PDF for each figure
- CLI: `--plot-all`, `--plot-best-vs-count`, `--plot-heatmap`, `--plot-bar`,
  `--output-dir`, `--fmt`

## Key results / observations

### Formulation comparison (10 shapes x 3 noise = 30 runs, alpha=1.0, 50 steps)
| Method           | RMSE_final | NIS_post | ms/step |
|------------------|-----------|----------|---------|
| Quat numeric     | ~3.4e-02  | ~10.5    | ~57     |
| Quat analytic ref| ~3.6e-02  | ~10.5    | ~34     |
| SO(3) analytic   | ~2.8e-02  | ~10.4    | ~29     |
| SO(3) numeric    | ~2.6e-02  | ~10.4    | ~56     |
- SO(3) analytic is ~2x faster than numeric for 5-D state (29 vs 56 ms/step)
- NIS_post ~ DOF = 9 for all methods, confirming filter is well-tuned at alpha=1.0

### Sensor placement study (10 shapes x 3 noise = 30 runs, alpha=1.0, 50 steps)
Selected best configs:
- 1 IMU: [0.25]          RMSE = 4.15e-01
- 2 IMU: [0.50, 1.00]    RMSE = 9.49e-03
- 3 IMU: [0.25, 0.50, 1.00]   RMSE = 7.43e-03
- 4 IMU: [0.25, 0.50, 0.75, 1.00]  RMSE = 6.77e-03
- 5 IMU: [0.20, 0.40, 0.60, 0.80, 1.00]  RMSE = 6.05e-03
- Large jump from 1 to 2 sensors; diminishing returns from 3 onward

## Problems / blockers
- The full 2-IMU grid heatmap (e.g. 8x8 sweep) is not saved in the CSV —
  only the 6 predefined pairs. The plotting script builds a sparse 4x4 heatmap
  from those. A full grid sweep requires re-running with `--heatmap-grid N`.

## Next steps
- Run the full 2-IMU heatmap sweep (`--heatmap-grid 8`) and consider adding
  a saver so the grid data is preserved in JSON for plot-only reuse
- Write the sensor placement section of the manuscript using the finalized results
- Consider running with larger shape banks (--num-shapes 20+) for robustness

## Related commits
- See `git log --oneline` on branch `cleanup-tests`

## Example commands used
```powershell
# Formulation comparison
python meas_model_compare.py `
  --mode formulation_compare `
  --num-shapes 10 --num-noise-realizations 3 `
  --formulation-alpha 1.0

# Sensor placement study
python sensor_placement_study_final.py --no-heatmap

# Plotting (no re-simulation)
python plot_sensor_placement_results.py --plot-all --output-dir figures
```
