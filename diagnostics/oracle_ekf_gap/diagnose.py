"""One existing Oracle case and 5D compatibility check; not a paper reproduction.

Uses the original curve-fit objective/initialization/options. Run before.json
before a mechanics fix, then --reuse-fit before.json for the same fitted state.
All lengths are SI unless explicitly labelled mm; wrench order is [moment, force].
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shape_force_est_imu.crt import virtual_work as vw
from shape_force_est_imu.ekf import kinematics as ekf_fk

spec = importlib.util.spec_from_file_location(
    "original_curve_fit", ROOT / "scripts/force_est/force_est_curve_fit.py"
)
curve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curve)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direction_deg(a, b, eps=1e-8):
    if min(np.linalg.norm(a), np.linalg.norm(b)) <= eps:
        return None  # zero/near-zero estimate OR truth has undefined direction
    return float(np.degrees(np.arccos(np.clip(
        a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


def physical_pull_fd(m, L, routes, orders, eps=1e-6, n_int=200):
    s = np.linspace(0.0, 1.0, n_int + 1)

    def pull(state):
        kappa = np.array([vw.curvature_from_modal(x, state, *orders) for x in s])
        lengths = [np.trapezoid(np.linalg.norm(
            np.array([0., 0., L]) + np.cross(kappa, r), axis=1), s) for r in routes]
        return -np.array(lengths)  # reference lengths constant w.r.t. m

    eye = np.eye(len(m))
    return np.column_stack([(pull(m + eps * e) - pull(m - eps * e)) / (2 * eps)
                            for e in eye])


def pull_check(m, L, routes, orders):
    analytic = vw.cable_jacobian(m, routes, L, *orders)
    numeric = physical_pull_fd(m, L, routes, orders)
    return {
        "state": m.tolist(), "order": list(orders), "step": 1e-6,
        "quadrature_intervals": 200, "units": "m per normalized modal coefficient",
        "max_abs_difference_m": float(np.max(np.abs(analytic - numeric))),
        "relative_frobenius_difference": float(np.linalg.norm(analytic - numeric)
                                               / np.linalg.norm(numeric)),
        "analytic": analytic.tolist(), "finite_difference": numeric.tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-fit", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {args.output}")

    dataset = ROOT / "artifacts/data/tdcr_gt_inext_2.npz"
    with np.load(dataset, allow_pickle=False) as data:
        meta = json.loads(data["meta"].item())
        T_gt, tau, f_gt, moment_gt = (data[k][0].copy() for k in ("T", "tau", "f_ext", "l_ext"))
        dataset_count = len(data["T"])
    L = meta["length_m"]
    orders, gamma = (2, 2, 1), 26
    s = np.linspace(0.0, 1.0, len(T_gt))
    config = {
        "dataset_sha256": sha256(dataset), "sample_index": 0,
        "order": list(orders), "gamma": gamma, "w_pos": 1.0, "w_rot": 1.0,
        "disk_step": 1, "optimizer": "L-BFGS-B", "maxiter": 200,
        "initial_state": [0.0] * 8, "L_m": L, "pinv_rcond": 1e-8,
        "E_Pa": meta["youngs_modulus"], "backbone_radius_m": meta["backbone_radius"],
        "tendon_offset_m": meta["tendon_offset"], "poisson": 0.3,
    }
    started = time.perf_counter()
    if args.reuse_fit:
        old = json.loads(args.reuse_fit.read_text(encoding="utf-8"))
        if old["config"] != config:
            raise ValueError("Cached fit has a different configuration")
        fit = old["fit"]
        m = np.array(fit["state"])
    else:
        result = minimize(curve.fit_cost, np.zeros(8),
                          args=(s, T_gt, gamma, L, 1.0, 1.0, *orders),
                          method="L-BFGS-B", options={"maxiter": 200})
        m = result.x
        fit = {
            "state": m.tolist(), "objective": float(result.fun),
            "success": bool(result.success), "status": int(result.status),
            "message": str(result.message), "nit": int(result.nit),
            "nfev": int(result.nfev), "elapsed_seconds": time.perf_counter() - started,
        }
    T_est = np.array([vw.product_of_exponentials(m, x, gamma, L, *orders) for x in s])
    dp = T_est[:, :3, 3] - T_gt[:, :3, 3]
    ang = np.array([curve.orientation_error_angle(a[:3, :3], b[:3, :3])
                    for a, b in zip(T_est, T_gt)])
    offset = meta["tendon_offset"]
    routes = [offset * np.array([np.cos(a), np.sin(a), 0.0])
              for a in np.linspace(0.0, 2 * np.pi, len(tau), endpoint=False)]
    I = np.pi * meta["backbone_radius"] ** 4 / 4
    EI = meta["youngs_modulus"] * I
    GJ = EI / (1 + config["poisson"])
    J_body, tip_est = vw.body_jacobian_at_s(m, 1.0, gamma, L, *orders)
    J_pull = vw.cable_jacobian(m, routes, L, *orders)
    gradU = vw.elastic_energy_gradient(m, EI, EI, GJ, L, *orders)
    body_wrench = vw.solve_wrench(J_body, J_pull, gradU, tau, 1e-8)
    world_tip_est = curve.wrench_body_tip_to_world_tip(body_wrench, tip_est)
    world_tip_gt = np.r_[moment_gt, f_gt]
    # Common physical reference: world axes, world/base origin. GT keeps its own geometry.
    base_est = curve.wrench_world_tip_to_world_base(world_tip_est, tip_est[:3, 3])
    base_gt = curve.wrench_world_tip_to_world_base(world_tip_gt, T_gt[-1, :3, 3])
    legacy_base_gt = curve.wrench_world_tip_to_world_base(world_tip_gt, tip_est[:3, 3])
    force_error = float(np.linalg.norm(world_tip_est[3:] - f_gt))
    moment_tip_error = float(np.linalg.norm(world_tip_est[:3] - moment_gt))
    metrics = {
        "information": "one static shape; noiseless full GT poses at 20 disks; no EKF/no IMU sequence",
        "sample_count": 1, "fit_position_sum_m2": float(np.sum(dp ** 2)),
        "fit_rotation_sum_rad2": float(np.sum(ang ** 2)),
        "shape_rms_mm": float(1000 * np.sqrt(np.mean(np.sum(dp ** 2, axis=1)))),
        "shape_mean_distance_mm": float(1000 * np.mean(np.linalg.norm(dp, axis=1))),
        "tip_error_mm": float(1000 * np.linalg.norm(dp[-1])),
        "orientation_rms_deg": float(np.degrees(np.sqrt(np.mean(ang ** 2)))),
        "force_error_N": force_error, "force_direction_deg": direction_deg(world_tip_est[3:], f_gt),
        "force_relative_magnitude_error": float(abs(np.linalg.norm(world_tip_est[3:])
                                                    - np.linalg.norm(f_gt)) / np.linalg.norm(f_gt)),
        "force_relative_vector_error": force_error / float(np.linalg.norm(f_gt)),
        "force_error_over_F0": force_error / (EI / L ** 2),
        "moment_error_world_tip_couples_Nm": moment_tip_error,
        "moment_direction_world_tip_deg": direction_deg(world_tip_est[:3], moment_gt),
        "moment_error_over_M0": moment_tip_error / (EI / L),
        "moment_error_common_world_base_Nm": float(np.linalg.norm(base_est[:3] - base_gt[:3])),
        "moment_error_legacy_gt_shift_Nm": float(np.linalg.norm(base_est[:3] - legacy_base_gt[:3])),
        "gt_shift_geometry_error_Nm": float(np.linalg.norm(base_gt[:3] - legacy_base_gt[:3])),
        "generalized_load_residual_norm_Nm": float(np.linalg.norm(
            J_body.T @ body_wrench - (gradU - J_pull.T @ tau))),
    }

    # Shared supported order (1,1,0): fixed nonzero state, no estimator substitution.
    m5 = np.array([0.6, -0.2, -0.4, 0.3, 0.25])
    pos_delta, rot_delta = [], []
    for x in np.linspace(0, 1, 21):
        a = vw.product_of_exponentials(m5, x, gamma, L, 1, 1, 0)
        b = ekf_fk.product_of_exponentials(m5, x, gamma=gamma, L=1000 * L, model="5d")
        pos_delta.append(np.linalg.norm(a[:3, 3] - b[:3, 3] / 1000))
        rot_delta.append(np.linalg.norm(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec()))
    fk_check = {"state": m5.tolist(), "point_count": 21, "gamma": gamma,
                "mechanics_L_m": L, "ekf_L_mm": 1000 * L,
                "max_position_difference_m": float(max(pos_delta)),
                "max_orientation_difference_rad": float(max(rot_delta))}
    checks = {
        "fk_5d": fk_check,
        "pull_5d": pull_check(m5, L, routes, (1, 1, 0)),
        "pull_fitted_8d": pull_check(m, L, routes, orders),
        "zero_direction_is_undefined": direction_deg(np.zeros(3), np.ones(3)) is None,
    }
    assert np.isclose(fit["objective"], metrics["fit_position_sum_m2"]
                      + metrics["fit_rotation_sum_rad2"], rtol=1e-10, atol=1e-14)
    assert max(pos_delta) < 1e-12 and max(rot_delta) < 1e-12
    assert checks["zero_direction_is_undefined"]
    tracked_sources = ["scripts/force_est/force_est_curve_fit.py",
                       "src/shape_force_est_imu/crt/virtual_work.py",
                       "src/shape_force_est_imu/ekf/kinematics.py"]
    output = {
        "scope": "current checkout compatibility diagnosis, not manuscript table reproduction",
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.executable, "numpy": np.__version__, "scipy": scipy.__version__,
        "source_sha256": {x: sha256(ROOT / x) for x in tracked_sources},
        "dataset": "artifacts/data/tdcr_gt_inext_2.npz", "dataset_count": dataset_count,
        "metadata": meta, "diagnostic_rng": "none; deterministic fit and fixed state",
        "config": config, "fit": fit, "fit_reused_from": str(args.reuse_fit) if args.reuse_fit else None,
        "metrics": metrics, "checks": checks,
        "stiffness": {"EI_Nm2": EI, "GJ_Nm2": GJ, "F0_N": EI / L ** 2, "M0_Nm": EI / L},
        "tau_N": tau.tolist(), "wrench_order": ["Mx", "My", "Mz", "Fx", "Fy", "Fz"],
        "world_tip_gt": world_tip_gt.tolist(), "world_tip_est": world_tip_est.tolist(),
        "world_base_gt": base_gt.tolist(), "world_base_est": base_est.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "fit": fit, "metrics": metrics,
                      "fk_5d": fk_check,
                      "pull_5d_relative_error": checks["pull_5d"]["relative_frobenius_difference"],
                      "pull_8d_relative_error": checks["pull_fitted_8d"]["relative_frobenius_difference"]}, indent=2))


if __name__ == "__main__":
    main()
