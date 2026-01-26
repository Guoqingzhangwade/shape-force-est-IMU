# shape-force-est-IMU 
 
Repo layout: 
- src/shape_force_est_imu: python package modules: crt, utils, io, core, viz 
- scripts: runnable scripts: figures, ekf, curve, crt, tools, notes 
- artifacts: generated outputs: data, figures 
- experiments: experiment-specific runs 
- data: datasets or raw inputs

Usage:
- Config-driven runs:
  - python scripts\run_config.py run-gt
  - python scripts\run_config.py run-est
  - python scripts\run_config.py run-both
  - Edit scripts\config\gt_config.json and scripts\config\est_config.json to change parameters.
- Task runner (predefined commands):
  - python scripts\run_tasks.py list
  - python scripts\run_tasks.py run config_run_both
- Example (inextensible GT + estimation):
  - python scripts\run_config.py run-both
  - python scripts\curve\3d_curve_fit_from_crt_gt.py --npz artifacts\data\tdcr_gt_inext_1.npz --sample-idx 0 --order-x 2 --order-y 2 --order-z 2 --plot-sample 0 --compute-wrench --plot-wrench-sample 0
