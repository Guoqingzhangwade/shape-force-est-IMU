"""Bounded A/B/C/D on original data; writes only to a fresh output directory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]

ORDER = (2, 2, 1)
GAMMA, STEPS, SIGMA_DEG, ALPHA, SEED = 10, 30, 0.5, 1.0, 1234


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def angle(a, b):
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def unique_case_index(case_ids, requested_id, label, parser):
    """Select by stored case ID independently in each dataset, never by row number."""
    case_ids = np.asarray(case_ids)
    if case_ids.ndim != 1 or case_ids.dtype.kind not in "iu":
        parser.error(f"{label} case_id must be a one-dimensional integer array")
    matches = np.flatnonzero(case_ids == requested_id)
    if len(matches) != 1:
        parser.error(f"Case ID {requested_id} must occur exactly once in {label}; found {len(matches)} matches")
    return int(matches[0])


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
        "--case-id", type=int, default=0,
        help="Stored physical case ID, resolved uniquely and independently in GT and IMU arrays (default: 0).",
    )
    parser.add_argument(
        "--noise-realization", type=int, default=0,
        help="Zero-based saved IMU noise index (default: 0; current dataset has five, indices 0 through 4).",
    )
    args = parser.parse_args()
    runner_path = Path(__file__).resolve()
    runner_sha256 = sha(runner_path)
    recovered = args.repo_root.expanduser().resolve()
    CRT = recovered / "src/shape_force_est_imu/crt"
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        parser.error(f"Output directory must be new: {args.output_dir}")
    source_path = CRT / "compare_oracle_vs_ekf_wrench_estimation.py"
    if not source_path.is_file():
        parser.error(f"Recovered source not found: {source_path}; select its repository with --repo-root")
    sys.path.insert(0, str(CRT))
    import compare_oracle_vs_ekf_wrench_estimation as original
    import virtual_work as vw
    assert Path(original.__file__).resolve() == source_path
    assert Path(vw.__file__).resolve() == CRT / "virtual_work.py"
    result_path = args.output_dir / "results.json"
    source_paths = ["compare_oracle_vs_ekf_wrench_estimation.py", "evaluate_kirchhoff_shape_estimation.py",
                    "virtual_work.py", "cosserat_rod_model.py", "evaluate_kirchhoff_wrench_constrained.py",
                    "generate_kirchhoff_gt_dataset_constrained.py"]
    input_paths = ["gt_data/kirchhoff_gt_dataset.npz", "gt_data/kirchhoff_imu_3imu.npz"]
    hashes = {name: sha(CRT / name) for name in source_paths + input_paths}
    production_source = source_path.read_text(encoding="utf-8")
    frechet_expression = "dePsi_dm[k][:, :, i] = expm_frechet(Psi_k, dP, compute_expm=False)"
    legacy_expression = "dePsi_dm[k][:, :, i] = ePsi_k @ (dP + 0.5 * (Psi_k @ dP - dP @ Psi_k))"
    if frechet_expression in production_source and legacy_expression not in production_source:
        derivative_implementation = "scipy.linalg.expm_frechet"
    elif legacy_expression in production_source and frechet_expression not in production_source:
        derivative_implementation = "legacy_truncated_commutator"
    else:
        derivative_implementation = "unrecognized_or_mixed_source; inspect recorded source hash"
    git_args = ["git", "-C", str(recovered), "-c", "core.fsmonitor=false"]
    status_args = git_args + ["status", "--porcelain=v1", "--untracked-files=normal", "--", "."]
    # New results may live inside this repository after migration; exclude only
    # this run's fresh directory when checking that preexisting work is unchanged.
    try:
        output_relative = args.output_dir.relative_to(recovered)
    except ValueError:
        output_relative = None
    if output_relative is not None:
        status_args.append(f":(exclude,literal){output_relative.as_posix()}")
    status_before = subprocess.check_output(status_args, text=True)
    with np.load(CRT / input_paths[0], allow_pickle=False) as d:
        gt_case_index = unique_case_index(d["case_id"], args.case_id, "GT dataset", parser)
        gt_meta = json.loads(d["meta"].item())
        gt_count = len(d["case_id"])
        gt_keys = ("positions", "orientations", "tau", "f_ext", "l_ext")
        if any(d[k].ndim == 0 or d[k].shape[0] != gt_count for k in gt_keys):
            parser.error("GT array case dimensions do not match its case_id array")
        if gt_meta["n_samples"] != gt_count:
            parser.error("GT metadata n_samples does not match its case_id array")
        pos, rotations, tau, force, moment = [d[k][gt_case_index].copy() for k in gt_keys]
        case_id = int(d["case_id"][gt_case_index])
        if pos.shape != (gt_meta["num_points"], 3) or rotations.shape != (len(pos), 9):
            parser.error("Selected GT position/orientation shapes do not match the saved metadata")
    with np.load(CRT / input_paths[1], allow_pickle=False) as d:
        imu_case_index = unique_case_index(d["case_id"], args.case_id, "IMU dataset", parser)
        imu_meta = json.loads(d["meta"].item())
        imu_count, n_imu = len(d["case_id"]), int(imu_meta["n_imu"])
        if d["R_true"].shape != (imu_count, n_imu, 3, 3):
            parser.error("IMU R_true shape does not match its case IDs and sensor metadata")
        if d["R_meas"].ndim != 5 or d["R_meas"].shape[0] != imu_count or d["R_meas"].shape[2:] != (n_imu, 3, 3):
            parser.error("IMU R_meas must have shape (case, noise, sensor, 3, 3)")
        noise_count = d["R_meas"].shape[1]
        if imu_meta["n_cases"] != imu_count or imu_meta["num_noise_realizations"] != noise_count:
            parser.error("IMU metadata case/noise counts do not match its arrays")
        if not 0 <= args.noise_realization < noise_count:
            parser.error(f"Noise realization must be between 0 and {noise_count - 1}; got {args.noise_realization}")
        truth_R = d["R_true"][imu_case_index].copy()
        noisy_R = d["R_meas"][imu_case_index, args.noise_realization].copy()
        nominal_s, imu_index = d["imu_positions"].copy(), d["imu_arc_indices"].copy()
        if nominal_s.shape != (n_imu,) or imu_index.shape != (n_imu,):
            parser.error("IMU sensor position/index shapes do not match n_imu")
        if imu_index.dtype.kind not in "iu" or np.any(imu_index < 0) or np.any(imu_index >= len(pos)):
            parser.error("IMU arc indices must be valid integer indices into the selected GT case")
        if imu_meta["num_pts_per_case"] != len(pos):
            parser.error("IMU metadata num_pts_per_case does not match the selected GT case")
        assert case_id == int(d["case_id"][imu_case_index])
    L = gt_meta["length_m"]
    actual_s = np.array(imu_meta["imu_actual_s"])
    if actual_s.shape != nominal_s.shape:
        parser.error("IMU metadata imu_actual_s must have one value per sensor")
    assert np.allclose(actual_s, imu_index / (len(pos) - 1))
    assert np.allclose(truth_R, rotations.reshape(-1, 3, 3)[imu_index])
    assert L == original._L and gt_meta["youngs_modulus"] == original._E
    assert gt_meta["backbone_radius"] == original._R_BB and gt_meta["tendon_offset"] == original._R_TENDON
    assert imu_meta["meas_std_deg"] == SIGMA_DEG
    sigma = np.deg2rad(SIGMA_DEG)
    s_all = np.linspace(0, 1, len(pos))
    # Validate selection, metadata, and mappings before creating any run output.
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def residual(m, observations):
        return np.concatenate([Rotation.from_matrix(
            R @ original.fwd_transform_general(m, float(s), GAMMA, L, *ORDER)[:3, :3].T
        ).as_rotvec() for s, R in zip(actual_s, observations)])

    m_oracle = original.estimate_modal_oracle(pos, rotations, *ORDER, L_phys=L)
    # Preserve the selected production EKF before offline fitting; no options substitution.
    t0 = time.perf_counter()
    m_ekf = original.run_ekf_for_order(noisy_R, actual_s, *ORDER, gamma=GAMMA,
                                      meas_std_deg=SIGMA_DEG, alpha=ALPHA, steps=STEPS)
    original_elapsed = time.perf_counter() - t0
    np.savez_compressed(args.output_dir / "original_states.npz", oracle=m_oracle, ekf=m_ekf)
    print("Dense-oracle and selected production EKF states saved.", flush=True)

    # A few fixed, GT-independent initializations. Preserve every convergence outcome.
    rng = np.random.default_rng(SEED)
    starts = [np.zeros(8), 0.01 * rng.standard_normal(8), 0.1 * rng.standard_normal(8)]
    trials, fitted = {}, {}
    for method, observations in [("B_noiseless_sparse", truth_R), ("C_noisy_sparse", noisy_R)]:
        trials[method] = []
        for start_idx, m0 in enumerate(starts):
            t0 = time.perf_counter()
            fit = least_squares(lambda m: residual(m, observations) / sigma, m0,
                                method="trf", max_nfev=120, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            record = {"start_index": start_idx, "initial_state": m0.tolist(), "state": fit.x.tolist(),
                      "success": bool(fit.success), "status": int(fit.status), "message": str(fit.message),
                      "nfev": int(fit.nfev), "njev": int(fit.njev), "cost_scaled": float(fit.cost),
                      "residual_norm_rad": float(np.linalg.norm(fit.fun) * sigma),
                      "elapsed_seconds": time.perf_counter() - t0}
            trials[method].append(record)
            print(method, "start", start_idx, "success", fit.success,
                  "residual", record["residual_norm_rad"], flush=True)
        best = min(trials[method], key=lambda row: row["cost_scaled"])
        fitted[method] = np.array(best["state"])

    def evaluate(m, method):
        T_all = np.array([original.fwd_transform_general(m, float(s), GAMMA, L, *ORDER) for s in s_all])
        dp = T_all[:, :3, 3] - pos
        ori = Rotation.from_matrix(T_all[:, :3, :3] @ rotations.reshape(-1, 3, 3).transpose(0, 2, 1)).as_rotvec()
        f_est, moment_est = original.wrench_from_modal(m, tau, *ORDER, gamma=GAMMA)
        wrench_tip = np.r_[moment_est, f_est]
        base_est = np.r_[moment_est + np.cross(T_all[-1, :3, 3], f_est), f_est]
        base_gt = np.r_[moment + np.cross(pos[-1], force), force]
        J, T_tip = vw.body_jacobian_at_s(m, 1.0, GAMMA, L, *ORDER)
        pull = vw.pull_jacobian(m, original._R_LIST, L, *ORDER)
        b = vw.generalized_modal_load(vw.elastic_energy_gradient(m, original._EIX,
               original._EIY, original._GJ, L, *ORDER), pull, tau)
        # World/base-origin wrench -> estimated body/tip wrench, with each own geometry.
        R, p = T_tip[:3, :3], T_tip[:3, 3]
        conversion = np.block([[R.T, -R.T @ vw.hat(p)], [np.zeros((3, 3)), R.T]])
        common_A = J.T @ conversion
        terms[method] = (common_A, b, base_est)
        index20 = np.round(np.linspace(0, len(pos) - 1, 20)).astype(int)
        metrics = {"method": method, "sample_count": 1, "state": m.tolist(),
            "evaluation_orientation_triplets": 1, "fit_residual_noisy_norm_rad": float(np.linalg.norm(residual(m, noisy_R))),
            "fit_residual_noiseless_norm_rad": float(np.linalg.norm(residual(m, truth_R))),
            "shape_rms_all100_mm": float(1000 * np.sqrt(np.mean(np.sum(dp ** 2, axis=1)))),
            "shape_mean_all100_mm": float(1000 * np.mean(np.linalg.norm(dp, axis=1))),
            "historical_rms_field_mean20_mm": float(1000 * np.mean(np.linalg.norm(dp[index20], axis=1))),
            "tip_mm": float(1000 * np.linalg.norm(dp[-1])),
            "orientation_rms_all100_deg": float(np.degrees(np.sqrt(np.mean(np.sum(ori ** 2, axis=1))))),
            "force_error_N": float(np.linalg.norm(f_est - force)), "force_direction_deg": angle(f_est, force),
            "force_error_over_F0": float(np.linalg.norm(f_est - force) / (original._EIX / L ** 2)),
            "historical_nrmse_field_relative_vector": float(np.linalg.norm(f_est - force) / np.linalg.norm(force)),
            "force_relative_magnitude_error": float(abs(np.linalg.norm(f_est) - np.linalg.norm(force)) / np.linalg.norm(force)),
            "moment_error_world_tip_Nm": float(np.linalg.norm(moment_est - moment)),
            "moment_error_common_world_base_Nm": float(np.linalg.norm(base_est[:3] - base_gt[:3])),
            "moment_direction_world_tip_deg": angle(moment_est, moment),
            "generalized_residual_norm_Nm": float(np.linalg.norm(b - common_A @ base_est)),
            "world_tip_wrench": wrench_tip.tolist(), "world_base_wrench": base_est.tolist()}
        if method.startswith("A_"):
            metrics["information"] = "100 dense noiseless GT orientations; adjacent SO(3) differences and axiswise linear LS; positions excluded from fitting"
        elif method.startswith("B_"):
            metrics["information"] = "one noiseless orientation triplet; 3 GT-independent starts; no prior"
        elif method.startswith("C_"):
            metrics["information"] = "one saved noisy orientation triplet, same as D; 3 GT-independent starts; no prior"
        else:
            metrics["information"] = ("same saved noisy triplet reused 30 updates; zero initialization; "
                                      "selected production P0/Q and analytic H; derivative=" + derivative_implementation)
        return metrics

    terms = {}
    rows = [evaluate(m_oracle, "A_dense_oracle"), evaluate(fitted["B_noiseless_sparse"], "B_noiseless_sparse"),
            evaluate(fitted["C_noisy_sparse"], "C_noisy_sparse"), evaluate(m_ekf, "D_production_ekf")]
    # Both sides of the wrench equation depend on shape; compare at a common reference.
    A0, b0, w0 = terms["A_dense_oracle"]
    for row in rows[1:]:
        A, b, w = terms[row["method"]]
        row["delta_b_norm_Nm"] = float(np.linalg.norm(b - b0))
        row["delta_A_times_oracle_wrench_norm_Nm"] = float(np.linalg.norm((A - A0) @ w0))
        row["net_generalized_perturbation_norm_Nm"] = float(np.linalg.norm(b - b0 - (A - A0) @ w0))

    def dense_curvature_residual(m):
        Rgt = rotations.reshape(-1, 3, 3)
        smid = (s_all[1:] + s_all[:-1]) / 2
        observed = Rotation.from_matrix(Rgt[:-1].transpose(0, 2, 1) @ Rgt[1:]).as_rotvec() / (s_all[1] - s_all[0])
        predicted = np.array([vw.curvature_from_modal(s, m, *ORDER) for s in smid])
        return float(np.linalg.norm(predicted - observed)), float(np.sqrt(np.mean((predicted - observed) ** 2)))
    oracle_norm, oracle_rms = dense_curvature_residual(m_oracle)
    fk_delta = [np.max(np.abs(original.fwd_transform_general(m_oracle, float(s), GAMMA, L, *ORDER)
                - vw.product_of_exponentials(m_oracle, float(s), GAMMA, L, *ORDER))) for s in actual_s]
    assert max(fk_delta) < 1e-12
    assert all(sha(CRT / name) == value for name, value in hashes.items())
    assert sha(runner_path) == runner_sha256
    status_after = subprocess.check_output(status_args, text=True)
    assert status_before == status_after
    result = {"scope": "one recovered fixed physical sample; no aggregate table rerun",
              "command_argv": list(sys.orig_argv),
              "runner": {"path": str(runner_path), "sha256": runner_sha256},
              "recovered_repo": str(recovered),
              "branch": subprocess.check_output(git_args + ["rev-parse", "--abbrev-ref", "HEAD"], text=True).strip(),
              "output_directory": str(args.output_dir),
              "status_check_excludes_output_directory": str(output_relative) if output_relative is not None else None,
              "head": subprocess.check_output(git_args + ["rev-parse", "HEAD"], text=True).strip(),
              "preexisting_status": status_before, "recovered_repo_status_preserved": True,
              "hashes": hashes, "production_derivative_implementation": derivative_implementation,
              "case_index": gt_case_index, "gt_case_index": gt_case_index,
              "imu_case_index": imu_case_index, "case_id": case_id,
              "noise_realization": args.noise_realization,
              "gt_metadata": gt_meta, "imu_metadata": imu_meta, "order": list(ORDER),
              "nominal_s": nominal_s.tolist(), "actual_s_used": actual_s.tolist(),
              "reason_for_actual_s": "manuscript table generator uses meta imu_actual_s; original compare CLI uses nominal positions",
              "gamma": GAMMA, "steps": STEPS, "sigma_deg": SIGMA_DEG, "alpha": ALPHA,
              "offline_seed": SEED, "offline_prior": None, "offline_max_nfev": 120,
              "original_ekf_elapsed_seconds": original_elapsed,
              "original_P0_diag": list(original.P0_DIAG) + [0.25] * 3,
              "original_Q_diag": list(original.Q_DIAG) + [original.Q_DIAG[1]] * 3,
              "oracle_fit": {"objective": "linear LS of curvature from adjacent GT orientation differences",
                             "initialization": "none; linear least squares", "convergence": "direct linear solution",
                             "residual_norm_normalized_curvature": oracle_norm,
                             "residual_component_rms_normalized_curvature": oracle_rms},
              "wrench_settings": {"L_m": L, "EI_Nm2": original._EIX, "GJ_Nm2": original._GJ,
                                   "pinv_rcond": 1e-8, "order": ["Mx", "My", "Mz", "Fx", "Fy", "Fz"],
                                   "tau_N": tau.tolist(), "gt_world_tip": np.r_[moment, force].tolist()},
              "trials": trials, "metrics": rows, "fk_max_abs_difference": float(max(fk_delta))}
    result_path.write_text(json.dumps(result, indent=2, default=lambda x: float(x)) + "\n", encoding="utf-8")
    fields = ["method", "sample_count", "information", "fit_residual_noisy_norm_rad",
              "fit_residual_noiseless_norm_rad", "shape_rms_all100_mm", "historical_rms_field_mean20_mm",
              "tip_mm", "force_error_N", "force_direction_deg", "force_error_over_F0",
              "moment_error_world_tip_Nm", "moment_error_common_world_base_Nm"]
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps([{k: row[k] for k in fields if k != "information"} for row in rows], indent=2), flush=True)


if __name__ == "__main__":
    main()
