# Progress Note — 2026-02-17

## What I did

### 1. EKF measurement model selection (manuscript subsection)
- Compared 4 EKF measurement model formulations: quaternion numeric/analytic, SO(3) numeric/analytic
- Ran Monte Carlo trials (N=10, random shape configurations) using `scripts/shape_est/meas_model_compare.py`
- Decided to show all 4 methods in the main text (no appendix) to justify the SO(3) analytic choice
- Generated publication figures using `scripts/shape_est/gen_comparison_figs.py`
- Fixed figure display issue: added auto log/linear scale detection when RMSE gap is >10×

### 2. Fixed make_fig1_hybrid.py (now fig1_shape_ekf_error.py)
- Broken data paths after file movements → fixed using project-root-relative path detection
- Unicode encoding error on Windows (σ, →) → replaced with ASCII-safe equivalents
- Results: Position error 1.078 ± 0.428 mm, Orientation error 1.461 ± 0.845 deg (N=10 samples)

### 3. Repository cleanup — archived obsolete scripts
- Moved 43 obsolete/experimental scripts to `scripts/archive/` and `src/shape_force_est_imu/crt/archive/`
  using `git mv` to preserve history
- Kept 10 active scripts

### 4. Renamed folders and scripts to descriptive names
- `scripts/ekf/`     → `scripts/shape_est/`
- `scripts/curve/`   → `scripts/force_est/`
- `scripts/crt/`     → `scripts/robot_sim/`
- `scripts/figures/` → `scripts/viz/`
- Individual scripts renamed (e.g., `EKF_shape_estimation_so3.py` → `shape_ekf_so3_demo.py`)
- `scripts/ekf/pub_figures/` → `results/meas_model_compare/` (generated outputs moved out of scripts)

### 5. CRT model consolidation
- `cosserat_rod_model_inextensible.py` → renamed to `cosserat_rod_model.py` (inextensible is the default)
- Old extensible-only `cosserat_rod_model.py` → archived as `archive/cosserat_rod_model_extensible_only.py`
- Simplified conditional import in `crt_gt_gen_output_file.py`

### 6. General housekeeping
- Deleted 8 empty/stale folders: `CRT_model/`, `scripts/ekf/`, `scripts/curve/`, `scripts/crt/`,
  `scripts/figures/`, `scripts/legacy/`, `data/`, `experiments/`
- Created `.gitignore` (`__pycache__/`, `.venv/`, stray root-level figures)
- Rewrote root `README.md` with current directory tree and updated usage commands
- Updated `scripts/README.md` with accurate script table and generated results locations

## Key results / observations
- SO(3) analytic EKF: RMSE ~0.030, computationally efficient (0.055 s/step)
- Quaternion analytic EKF: RMSE ~0.532 (diverges — not suitable)
- Shape EKF tip pose error: position 1.078 ± 0.428 mm, orientation 1.461 ± 0.845 deg

## Problems / blockers
- None currently

## Next steps
- Run baseline comparison experiments (main validation section of the paper)
- Validate force/wrench estimation pipeline end-to-end
- Generate final publication figures for manuscript submission (IEEE TRO)

## Related commits
- `58330e8` Fix make_fig1_hybrid.py paths and encoding issues
- `455f581` Archive obsolete scripts, keep only active ones
- `5e2c204` Rename folders and scripts to descriptive names
- `7531513` Update scripts/README.md with current folder structure
- `9ef5482` Clean up empty folders, stale files, and add .gitignore
- `0eecc36` Make inextensible Cosserat rod model the primary model
- `11abadc` Rewrite README with current project structure and usage
