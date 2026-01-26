# Progress Notes

Date: Today

Summary:
- Added config-driven workflow for GT generation and estimation (run_config.py + JSON configs).
- Added inextensible Cosserat model option and wired generator to switch models with --inextensible.
- Added scale-consistent wrench ranges (use_scaled_wrench, c_f, c_m) and normalized wrench metrics.
- Added warning checks when estimator parameters (E, backbone_radius, tendon_offset) mismatch GT meta.
- Added plotting for all wrench samples and equal-axis scaling in shape plots.

Key files:
- scripts/run_config.py, scripts/config/gt_config.json, scripts/config/est_config.json
- src/shape_force_est_imu/crt/crt_gt_gen_output_file.py
- scripts/curve/3d_curve_fit_from_crt_gt.py
- src/shape_force_est_imu/crt/cosserat_rod_model_inextensible.py

Common commands:
- Generate GT from config:
  - python scripts\run_config.py run-gt
- Run estimation from config:
  - python scripts\run_config.py run-est
- Run both:
  - python scripts\run_config.py run-both
- Sweep estimator E only (mismatch study):
  - python scripts\run_config.py sweep-est --scales 0.5,0.8,1.0,1.2,1.5

Notes:
- Estimation uses tau from GT data; only GT generation uses tau_max.
- Estimator should match GT parameters (E, backbone_radius, tendon_offset, length) to avoid warnings.
- Inextensible model still shows GT-vs-virtual-work mismatch; Jacobians are consistent by FD checks.

Next steps:
- Run multi-sample estimation with plot_wrench_all for visual inspection.
- If needed, add sweep-both (regenerate GT per E) or routing-radius sensitivity sweep.
