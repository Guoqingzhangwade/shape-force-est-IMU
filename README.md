# shape-force-est-IMU

Shape and force estimation for tendon-driven continuum robots (TDCR) using IMU measurements
and an Extended Kalman Filter (EKF) on SO(3).

---

## Repository Structure

```
shape-force-est-IMU/
│
├── src/shape_force_est_imu/      # Importable Python library
│   ├── crt/                      # Cosserat rod theory (CRT) model & data generation
│   │   ├── cosserat_rod_model.py     # CRT forward kinematics (inextensible/Kirchhoff)
│   │   ├── virtual_work.py           # Wrench/force estimation via virtual work
│   │   ├── crt_gt_gen.py             # Ground-truth shape generation (in-memory)
│   │   ├── crt_gt_gen_output_file.py # Ground-truth generation → .npz file
│   │   ├── crt_meas_gen_output_file.py # Simulated IMU measurements → .npz file
│   │   ├── generator.py              # Sample/model generation helpers
│   │   ├── pck_models.py             # PCK0/PCK2 model utilities
│   │   └── utils.py                  # Math helpers (hat, unit, etc.)
│   ├── ekf/                      # EKF library modules (SO3/quat, Jacobians)
│   ├── core/, io/, utils/, viz/  # Other library submodules
│
├── scripts/                      # Runnable scripts
│   ├── shape_est/                # Shape estimation (EKF-based)
│   │   ├── shape_ekf_so3_demo.py     # Single-sample EKF demo (SO(3))
│   │   ├── shape_ekf_5d.py           # 5D polynomial shape EKF
│   │   ├── meas_model_compare.py     # Compare 4 measurement model formulations
│   │   ├── gen_comparison_figs.py    # Generate publication figures
│   │   └── README.md                 # Figure generation guide
│   ├── force_est/                # Force estimation
│   │   └── force_est_curve_fit.py    # Force estimation via curve fitting from CRT GT
│   ├── robot_sim/                # TDCR robot simulation
│   │   └── tdcr_sim.py               # TDCR kinematics / forward simulation
│   ├── viz/                      # Visualization
│   │   ├── fig1_shape_ekf_error.py   # Fig. 1: violin+box plot of tip pose error
│   │   └── plot_shape_with_cov.py    # Estimated shapes with covariance ellipsoids
│   ├── config/                   # Config files for run_config.py
│   │   ├── gt_config.json
│   │   └── est_config.json
│   ├── tools/
│   │   └── check_encoding.py         # Scan repo files for encoding issues
│   ├── run_config.py             # Config-driven pipeline runner
│   ├── run_tasks.py              # Predefined task runner
│   └── archive/                  # Obsolete scripts (preserved, not intended to run)
│
├── results/                      # Generated output (figures, tables)
│   ├── meas_model_compare/       # Output of gen_comparison_figs.py
│   └── shape_est/                # Output of fig1_shape_ekf_error.py
│
├── artifacts/                    # Data artifacts
│   ├── data/                     # Ground-truth & measurement .npz datasets
│   └── figures/                  # Other saved figures
│
├── notes/                        # Development notes
├── progress/                     # Session progress notes (one .md file per work session)
│   └── TEMPLATE.md               # Copy this to start a new entry
└── .gitignore
```

---

## Usage

### Config-driven pipeline (generate GT data + run estimation)

```bash
# Edit parameters first:
#   scripts/config/gt_config.json   — ground-truth generation settings
#   scripts/config/est_config.json  — estimation settings

python scripts/run_config.py run-gt      # Generate ground-truth data
python scripts/run_config.py run-est     # Run estimation
python scripts/run_config.py run-both    # Both steps in sequence
```

### Task runner (predefined commands)

```bash
python scripts/run_tasks.py list
python scripts/run_tasks.py run config_run_both
```

### Shape EKF — measurement model comparison

```bash
cd scripts/shape_est
python meas_model_compare.py            # Run Monte Carlo comparison
python gen_comparison_figs.py           # Generate publication figures
```

### Force estimation from CRT ground truth

```bash
python scripts/force_est/force_est_curve_fit.py \
  --npz artifacts/data/tdcr_gt_inext_1.npz \
  --sample-idx 0 --order-x 2 --order-y 2 --order-z 2 \
  --plot-sample 0 --compute-wrench --plot-wrench-sample 0
```

### Fig. 1 — Tip pose error (violin+box plot)

```bash
python scripts/viz/fig1_shape_ekf_error.py
# Output: results/shape_est/fig1_pose_error_10.pdf
```
