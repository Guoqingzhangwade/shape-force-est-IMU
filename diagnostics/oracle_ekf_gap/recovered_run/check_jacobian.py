"""Bounded pre-fix versus Frechet derivative comparison for the recovered 8D EKF.

Run with -B. Source and inputs are read-only; output requires a fresh directory.
The baseline and fixed EKF receive identical case 0/noise 0 observations and
actual stored sensor positions. A fixed production source requires --baseline-ref;
the old baseline is then loaded from Git and executed only in memory.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import scipy


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_source(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def memory_module(source, name, source_path):
    module = types.ModuleType(name)
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path) + f" [{name}]", "exec"), module.__dict__)
    return module


def direction_deg(estimate, truth):
    if min(np.linalg.norm(estimate), np.linalg.norm(truth)) <= 1e-8:
        return None
    cosine = estimate @ truth / (np.linalg.norm(estimate) * np.linalg.norm(truth))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def capture_closures(module, observations, positions):
    captured = {}

    def trace(frame, event, arg):
        if frame.f_code is module.run_ekf_for_order.__code__ and event == "return":
            for name in ("H_and_r_general", "compute_dR_dm_general", "fwd_rotation_general"):
                captured[name] = frame.f_locals[name]
        return trace

    previous_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        module.run_ekf_for_order(observations, positions, 2, 2, 1,
                                 gamma=10, meas_std_deg=0.5, alpha=1.0, steps=0)
    finally:
        sys.settrace(previous_trace)
    return captured


def jacobian_check(closure, state):
    residual_function = closure["H_and_r_general"]
    analytic, residual = residual_function(state)
    epsilon = 1e-6
    numeric = np.column_stack([
        (residual_function(state + epsilon * e)[1]
         - residual_function(state - epsilon * e)[1]) / (2 * epsilon)
        for e in np.eye(state.size)
    ])
    return {
        "state": state.tolist(), "central_difference_step": epsilon,
        "residual_convention": "Log(R_meas @ R_pred.T); additive modal perturbations",
        "residual_norm_rad": float(np.linalg.norm(residual)),
        "analytic": analytic.tolist(), "finite_difference": numeric.tolist(),
        "max_abs_difference": float(np.max(np.abs(analytic - numeric))),
        "relative_frobenius_difference": float(np.linalg.norm(analytic - numeric)
                                                / np.linalg.norm(numeric)),
    }


def metrics(module, state, observations, positions, tau, f_true, l_true,
            gt_positions, gt_tip_position):
    force, moment = module.wrench_from_modal(state, tau, 2, 2, 1, gamma=10)
    transforms = np.array([
        module.fwd_transform_general(state, float(s), 10, 0.1, 2, 2, 1)
        for s in np.linspace(0.0, 1.0, len(gt_positions))
    ])
    estimate_position = transforms[-1, :3, 3]
    world_base_moment_est = moment + np.cross(estimate_position, force)
    world_base_moment_gt = l_true + np.cross(gt_tip_position, f_true)
    residuals = [module.so3_log(R @ module.fwd_transform_general(
        state, float(s), 10, 0.1, 2, 2, 1)[:3, :3].T)
                 for R, s in zip(observations, positions)]
    world_tip_est = np.r_[moment, force]
    world_tip_gt = np.r_[l_true, f_true]
    sigma = np.deg2rad(0.5)
    return {
        "state": state.tolist(), "world_tip_est_moment_force": world_tip_est.tolist(),
        "world_tip_gt_moment_force": world_tip_gt.tolist(),
        "world_base_est_moment_force": np.r_[world_base_moment_est, force].tolist(),
        "world_base_gt_moment_force": np.r_[world_base_moment_gt, f_true].tolist(),
        "force_error_N": float(np.linalg.norm(force - f_true)),
        "force_direction_deg": direction_deg(force, f_true),
        "force_error_over_F0": float(np.linalg.norm(force - f_true) / (module._EIX / module._L**2)),
        "moment_error_world_tip_couples_Nm": float(np.linalg.norm(moment - l_true)),
        "moment_direction_world_tip_deg": direction_deg(moment, l_true),
        "moment_error_common_world_base_Nm": float(np.linalg.norm(world_base_moment_est - world_base_moment_gt)),
        "shape_rms_mm": float(1000 * np.sqrt(np.mean(np.sum(
            (transforms[:, :3, 3] - gt_positions)**2, axis=1)))),
        "orientation_observation_residual_norm_rad": float(np.linalg.norm(residuals)),
        "orientation_observation_least_squares_cost": float(0.5 * np.sum(np.array(residuals)**2) / sigma**2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", "--recovered-root", dest="repo_root", type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository containing the recovered CRT source and GT/IMU files; "
             "defaults to the repository containing this script (parents[3]).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="New directory for this run; must not already exist. Historical evidence is never replaced.",
    )
    parser.add_argument(
        "--baseline-ref",
        help="Explicit pre-fix Git commit or ref; required after production uses expm_frechet. "
             "The baseline is read with git show and never restored to the working tree.",
    )
    args = parser.parse_args()
    if not sys.dont_write_bytecode:
        raise RuntimeError("Run this diagnostic with python -B")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        parser.error(f"Output directory must be new: {output_dir}")
    output_path = output_dir / "jacobian_correction.json"
    patch_path = output_dir / "jacobian_correction.patch"
    root = args.repo_root.expanduser().resolve()
    crt = root / "src/shape_force_est_imu/crt"
    source_path = crt / "compare_oracle_vs_ekf_wrench_estimation.py"
    if not source_path.is_file():
        parser.error(f"Recovered source not found: {source_path}; select its repository with --repo-root")
    gt_path = crt / "gt_data/kirchhoff_gt_dataset.npz"
    imu_path = crt / "gt_data/kirchhoff_imu_3imu.npz"
    protected = [source_path, crt / "virtual_work.py",
                 crt / "evaluate_kirchhoff_shape_estimation.py",
                 crt / "cosserat_rod_model.py",
                 crt / "evaluate_kirchhoff_wrench_constrained.py",
                 crt / "generate_kirchhoff_gt_dataset_constrained.py",
                 gt_path, imu_path]
    hashes_before = {str(p.relative_to(root)): digest(p) for p in protected}
    source = normalize_source(source_path.read_text(encoding="utf-8"))
    old_import = "from scipy.linalg import block_diag as _block_diag"
    new_import = "from scipy.linalg import block_diag as _block_diag, expm_frechet"
    old_expression = "dePsi_dm[k][:, :, i] = ePsi_k @ (dP + 0.5 * (Psi_k @ dP - dP @ Psi_k))"
    new_expression = "dePsi_dm[k][:, :, i] = expm_frechet(Psi_k, dP, compute_expm=False)"
    baseline_commit = baseline_blob = None
    if args.baseline_ref:
        git_args = ["git", "-C", str(root)]
        baseline_commit = subprocess.check_output(
            git_args + ["rev-parse", "--verify", "--end-of-options", args.baseline_ref + "^{commit}"],
            text=True,
        ).strip()
        source_spec = baseline_commit + ":" + source_path.relative_to(root).as_posix()
        baseline_blob = subprocess.check_output(git_args + ["rev-parse", source_spec], text=True).strip()
        baseline_bytes = subprocess.check_output(git_args + ["show", source_spec])
        baseline_source = normalize_source(baseline_bytes.decode("utf-8"))
        mode = "pre_fix_git_baseline_vs_fixed_production"
    else:
        if new_expression in source:
            parser.error("Production already uses expm_frechet; supply --baseline-ref with the explicit pre-fix commit")
        baseline_bytes = source_path.read_bytes()
        baseline_source = source
        mode = "pre_fix_production_vs_in_memory_correction"
    if (baseline_source.count(old_import) != 1 or new_import in baseline_source
            or baseline_source.count(old_expression) != 1 or new_expression in baseline_source):
        parser.error("Selected baseline does not contain exactly the expected pre-fix import and derivative expression")
    corrected_source = baseline_source.replace(old_import, new_import, 1).replace(old_expression, new_expression, 1)
    if args.baseline_ref and source != corrected_source:
        parser.error("Production source does not exactly match the baseline plus the intended two-line Frechet fix (LF-normalized)")
    output_dir.mkdir(parents=True, exist_ok=False)
    patch = "".join(difflib.unified_diff(
        baseline_source.splitlines(keepends=True), corrected_source.splitlines(keepends=True),
        fromfile="a/src/shape_force_est_imu/crt/compare_oracle_vs_ekf_wrench_estimation.py",
        tofile="b/src/shape_force_est_imu/crt/compare_oracle_vs_ekf_wrench_estimation.py",
    ))
    sys.path.insert(0, str(crt))
    import compare_oracle_vs_ekf_wrench_estimation as production
    assert Path(production.__file__).resolve() == source_path
    if args.baseline_ref:
        baseline = memory_module(baseline_source, "pre_fix_ekf_git_baseline", source_path)
        fixed = production
        baseline_label, fixed_label = "pre_fix_baseline", "production_fixed"
    else:
        baseline = production
        fixed = memory_module(corrected_source, "recovered_ekf_in_memory_frechet_diagnostic", source_path)
        baseline_label, fixed_label = "original", "corrected_in_memory"

    with np.load(gt_path, allow_pickle=False) as gt:
        gt_positions = gt["positions"][0].copy()
        gt_orientations = gt["orientations"][0].copy()
        tau, f_true, l_true = (gt[key][0].copy() for key in ("tau", "f_ext", "l_ext"))
        gt_tip_position = gt["T_tip"][0, :3, 3].copy()
        gt_case_id = int(gt["case_id"][0])
        gt_meta = json.loads(gt["meta"].item())
    with np.load(imu_path, allow_pickle=False) as imu:
        observations = list(imu["R_meas"][0, 0].copy())
        imu_case_id = int(imu["case_id"][0])
        nominal_s = imu["imu_positions"].copy()
        actual_s = imu["imu_arc_indices"] / (len(gt_positions) - 1)
        imu_meta = json.loads(imu["meta"].item())
    assert gt_case_id == imu_case_id == 0
    assert np.allclose(actual_s, np.array([25 / 99, 50 / 99, 1.0]), rtol=0, atol=1e-15)
    assert gt_meta["length_m"] == baseline._L == production._L == 0.1
    oracle_state = baseline.estimate_modal_oracle(gt_positions, gt_orientations, 2, 2, 1)
    results = {}
    for label, module in ((baseline_label, baseline), (fixed_label, fixed)):
        closure = capture_closures(module, observations, actual_s)
        check = jacobian_check(closure, oracle_state)
        state = module.run_ekf_for_order(observations, actual_s, 2, 2, 1,
                                         gamma=10, meas_std_deg=0.5, alpha=1.0, steps=30)
        results[label] = {
            "jacobian_at_common_oracle_state": check,
            "final": metrics(module, state, observations, actual_s, tau, f_true, l_true,
                             gt_positions, gt_tip_position),
        }
    assert results[fixed_label]["jacobian_at_common_oracle_state"]["relative_frobenius_difference"] < 1e-6
    hashes_after = {str(p.relative_to(root)): digest(p) for p in protected}
    assert hashes_before == hashes_after, "Read-only recovered inputs or protected sources changed"
    output = {
        "scope": "one same-case derivative comparison; no production source modification by this runner",
        "comparison_mode": mode,
        "baseline_ref_requested": args.baseline_ref,
        "baseline_commit": baseline_commit,
        "baseline_blob": baseline_blob,
        "baseline_source_sha256_bytes": hashlib.sha256(baseline_bytes).hexdigest(),
        "normalized_source_sha256_utf8": {
            label: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for label, text in (("pre_fix_baseline", baseline_source),
                                ("fixed", corrected_source), ("production", source))
        },
        "production_matches_baseline_plus_two_line_fix": bool(args.baseline_ref),
        "baseline_dependency_scope": "Both modules use the same current checkout dependencies; this isolates the two-line derivative change, not an entire historical checkout.",
        "recovered_root": str(root),
        "output_directory": str(output_dir),
        "recovered_head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "command_argv": [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
        "python": sys.executable, "numpy": np.__version__, "scipy": scipy.__version__,
        "source_input_sha256_before": hashes_before, "source_input_sha256_after": hashes_after,
        "protected_source_and_input_hashes_unchanged": hashes_before == hashes_after,
        "changes": ["add expm_frechet import", "replace one exponential derivative expression"],
        "patch_direction": "pre-fix baseline to fixed implementation; recorded only, never applied by this runner",
        "preserved": "same residual, prior, process covariance, observation covariance, update, initialization, 30 steps and wrench inversion",
        "configuration": {
            "case_index": 0, "case_id": 0, "noise_realization_index": 0,
            "orders": [2, 2, 1], "gamma": 10, "meas_std_deg": 0.5,
            "alpha": 1.0, "steps": 30, "gt_seed": gt_meta["rng_seed"],
            "imu_seed": imu_meta["rng_seed"], "diagnostic_random_draws": "none",
            "nominal_sensor_positions": nominal_s.tolist(), "actual_sensor_positions": actual_s.tolist(),
            "P0_diag": [1, 0.5, 1, 0.5, 0.25, 0.25, 0.25, 0.25],
            "Q_diag": [1e-10, 1e-12, 1e-10, 1e-12, 1e-10, 1e-12, 1e-12, 1e-12],
            "information": "one fixed noisy frame re-used 30 times; static, not independent new measurements",
        },
        "results": results,
        "interpretation_limit": "A same-case difference isolates this derivative change; it does not reproduce an aggregate manuscript table or establish global observability.",
    }
    patch_path.write_text(patch, encoding="utf-8")
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({label: {
        "jacobian_relative_error": value["jacobian_at_common_oracle_state"]["relative_frobenius_difference"],
        "final": value["final"],
    } for label, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
