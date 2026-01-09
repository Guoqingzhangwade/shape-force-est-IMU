## Scripts and CRT Modules

This file is a quick map of what each script/module is for. Most scripts are
standalone and assume you run them from the repo root. Data inputs and outputs
often live under `artifacts/data` and `artifacts/figures`.

CRT module (`src/shape_force_est_imu/crt/`)
- cosserat_rod_model.py: Cosserat rod forward kinematics model for the CRT.
- crt_gt_gen.py: Generate CRT ground-truth shapes/poses (in-memory).
- crt_gt_gen_output_file.py: Generate and save CRT ground-truth datasets.
- crt_gt_gen_plot.py: Plot ground-truth CRT samples.
- crt_meas_gen_output_file.py: Generate and save simulated measurement data.
- CRT_time_comparison.py: Baseline runtime comparison of PCK vs CRT methods.
- CRT_time_comparison_6D_for_CRT_v2.py: Runtime comparison variant for 6D CRT.
- CRT_time_comparison_in_plane_assumption.py: Timing under in-plane assumption.
- CRT_time_comparison_relax_assumption.py: Timing with relaxed assumptions and
  inverse-wrench estimation.
- generator.py: Helper utilities for CRT sample/model generation.
- pck_models.py: PCK0/PCK2 model utilities and inverse/fit helpers.
- utils.py: CRT-specific math helpers.

Runnable scripts (`scripts/`)
Figures (`scripts/figures/`)
- make_fig1_hybrid.py: Run shape EKF on samples and draw Fig. 1 (hybrid plot).
- make_fig1_hybrid_debug.py: Debug version of Fig. 1 pipeline with extra checks.
- make_fig1_hybrid_debug_plot.py: Debug plotting helpers for Fig. 1 pipeline.
- make_fig1_hybrid_plot_all_debug.py: Plot-all debug variant for Fig. 1.
- plot_example_est_shapes.py: Plot example estimated shapes.
- plot_example_est_shapes_with_covariance.py: Plot shapes with covariance.
- plot_example_est_shapes_with_covariance_v2.py: Updated covariance plot variant.

Curve tests (`scripts/curve/`)
- 3d_curve_FK_Lie_group.py: 3D curve forward kinematics using Lie group form.
- 3d_curve_test.py: Baseline 3D curve test script.
- 3d_curve_test_9meas.py: 3D curve test using 9 measurements.
- 3d_curve_test_enhanced_inital.py: Curve test with enhanced initialization.
- 3d_curve_test_Magnus.py: Curve test using Magnus integration.
- 3d_curve_test_Magnus_v1.py: Magnus curve test variant v1.
- 3d_curve_test_Magnus_v2.py: Magnus curve test variant v2.
- 3d_curve_test_Magnus_v3.py: Magnus curve test variant v3.
- 3d_curve_test_Magnus_v4_constant_strain.py: Magnus test with constant strain.
- 3d_curve_test_Magnus_v5_linear.py: Magnus test with linear strain model.
- 3d_curve_test_new.py: New/experimental 3D curve test variant.

CRT scripts (`scripts/crt/`)
- 3d_CR_with_disc_tendon.py: 3D Cosserat rod with disk tendon model.
- 3d_CR_with_disc_tendon_v2.py: Updated disk tendon model (v2).
- 3d_CR_with_disc_tendon_v3.py: Updated disk tendon model (v3).
- 3d_CR_with_disc_tendon_plot_true_est.py: Plot true vs estimated shapes.
- 3d_CR_with_disc_tendon_sample_shape_gen.py: Sample shape generation utility.

EKF/IEKF (`scripts/ekf/`)
- curvature_estimation_commented.py: Curvature estimation (commented/annotated).
- EKF_shape_estimation_v0.py: EKF shape estimation (version 0).
- EKF_shape_estimation_v1.py: EKF shape estimation (version 1).
- EKF_shape_estimation_v2.py: EKF shape estimation (version 2).
- EKF_shape_estimation_v3.py: EKF shape estimation (version 3).
- EKF_shape_estimation_v4.py: EKF shape estimation (version 4).
- shape_EKF_5d.py: 5D shape EKF variant.
- shape_EKF_5d_v2.py: 5D shape EKF variant v2.
- shape_EKF_6d.py: 6D shape EKF variant.
- shape_EKF_6d_plot.py: Plotting helpers for 6D EKF results.
- shape_EKF_9d.py: 9D shape EKF variant.
- shape_IEKF_v0.py: Iterated EKF variant (v0).
- shape_IEKF_v1.py: Iterated EKF variant (v1).
- shape_est.py: Shape estimation utility/entry point.

Notes and tools
- scripts/notes/shape_ekf.txt: Notes on shape EKF development.
- scripts/tools/check_encoding.py: Scan repo files for encoding issues.

Legacy
- scripts/legacy/cosserat_rod_model.py: Legacy Cosserat rod model (pre-refactor).
