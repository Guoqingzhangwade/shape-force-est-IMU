# Scripts Overview

Quick map of every active script in this repository.
Obsolete/experimental scripts are moved to `scripts/archive/`.

---

## Active Scripts

| Path | Purpose |
|---|---|
| `scripts/shape_est/shape_ekf_so3_demo.py` | Single-sample EKF demo (SO(3) measurement model) |
| `scripts/shape_est/shape_ekf_5d.py` | 5D polynomial shape EKF |
| `scripts/shape_est/meas_model_compare.py` | Compare 4 measurement model formulations (Monte Carlo) |
| `scripts/shape_est/gen_comparison_figs.py` | Generate publication figures from comparison results |
| `scripts/shape_est/README.md` | Detailed guide for figure generation |
| `scripts/force_est/force_est_curve_fit.py` | Force estimation via curve fitting from CRT ground truth |
| `scripts/robot_sim/tdcr_sim.py` | TDCR robot kinematics / forward simulation |
| `scripts/viz/fig1_shape_ekf_error.py` | Fig. 1: hybrid violin+box plot of tip pose error |
| `scripts/viz/plot_shape_with_cov.py` | Plot estimated backbone shapes with covariance ellipsoids |
| `scripts/tools/check_encoding.py` | Scan repo files for encoding issues (Windows utility) |
| `scripts/run_tasks.py` | Task runner entry point |
| `scripts/run_config.py` | Configuration loader for task runner |

---

## Generated Results

| Path | Contents |
|---|---|
| `results/meas_model_compare/` | Figures and LaTeX tables from `gen_comparison_figs.py` |
| `results/shape_est/` | Output figures from `fig1_shape_ekf_error.py` |
| `artifacts/data/` | Ground-truth and measurement `.npz` datasets |
| `artifacts/figures/` | Other saved figures |

---

## Source Library (`src/`)

| Module | Purpose |
|---|---|
| `src/shape_force_est_imu/crt/cosserat_rod_model.py` | Cosserat rod forward kinematics |
| `src/shape_force_est_imu/crt/crt_gt_gen.py` | CRT ground-truth shape/pose generation (in-memory) |
| `src/shape_force_est_imu/crt/crt_gt_gen_output_file.py` | Save CRT ground-truth datasets to file |
| `src/shape_force_est_imu/crt/crt_meas_gen_output_file.py` | Generate and save simulated measurement data |
| `src/shape_force_est_imu/crt/virtual_work.py` | Wrench/force estimation via virtual work principle |
| `src/shape_force_est_imu/crt/pck_models.py` | PCK0/PCK2 model utilities |
| `src/shape_force_est_imu/crt/utils.py` | CRT math helpers |
| `src/shape_force_est_imu/crt/generator.py` | Sample/model generation helpers |

---

## Archive

Obsolete scripts are preserved in `scripts/archive/` (and `src/shape_force_est_imu/crt/archive/`)
with their git history intact. They are not intended to be run.
