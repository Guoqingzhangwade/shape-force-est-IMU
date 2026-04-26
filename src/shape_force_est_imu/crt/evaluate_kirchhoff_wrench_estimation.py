#!/usr/bin/env python3
"""
Cross-model robustness study - Step 4: Tip-wrench estimation evaluation.

Loads the Kirchhoff GT dataset (Step 1) and IMU measurements (Step 2),
re-runs the same SO(3)-analytic EKF (Step 3) to obtain modal coordinate
estimates m_est, then infers tip wrenches via two methods:

  1. Direct  -- least-norm pseudo-inverse: F̄ = pinv(J_Vb_m.T) @ b_w

  2. MAP     -- F̂ = argmin ||F - F̄||²_{Σ_F^{-1}} + ||F||²_{Q_F^{-1}}
               Fuses the direct estimate F̄ with a zero-mean prior using
               the EKF-propagated wrench covariance Σ_F (augmented with a
               cross-model uncertainty term σ_model) and the GT-range prior Q_F.

Both methods use the polynomial modal virtual-work equations (cross-model
mismatch relative to the Kirchhoff GT), mirroring the shape-estimation
cross-model study in Step 3.

Key empirical finding: force NRMSE ≈ 55%, moment NRMSE ≈ 11%.  The cross-model
mismatch dominates, so MAP with zero prior gives results similar to Direct.

Output files (in --save-dir)
----------------------------
  kirchhoff_wrench_est_results.csv        per-(case, noise_real, layout, method)
  kirchhoff_wrench_est_summary.csv        aggregated per (layout, method)
  kirchhoff_wrench_est_results.json       structured full results
  kirchhoff_wrench_est_data.npz           estimates + GT for plotting

Usage
-----
  cd src/shape_force_est_imu/crt
  python evaluate_kirchhoff_wrench_estimation.py
  python evaluate_kirchhoff_wrench_estimation.py --save-dir gt_data/results
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Shape-EKF utilities (re-used from Step 3 without modification)
# ---------------------------------------------------------------------------
from evaluate_kirchhoff_shape_estimation import (
    ALPHA_DEFAULT,
    E3,
    GAMMA_DEFAULT,
    MEAS_STD_DEG_DEF,
    P0_DIAG,
    Q_DIAG,
    STATE_DIM,
    STEPS_DEFAULT,
    load_ground_truth_dataset,
    load_imu_measurements,
    run_ekf_on_frame,
)

# ---------------------------------------------------------------------------
# Virtual-work / mechanics utilities
# ---------------------------------------------------------------------------
from virtual_work import (
    body_jacobian_at_s,
    elastic_energy_gradient,
    generalized_modal_load,
    gram_matrix,
    pull_jacobian,
    solve_wrench,
)
from scipy.linalg import block_diag as _block_diag

# ---------------------------------------------------------------------------
# Rod parameters (must match generate_kirchhoff_gt_dataset.py defaults)
# ---------------------------------------------------------------------------
_L       = 0.1          # physical length [m]
_E       = 60e9         # Young's modulus [Pa]
_NU      = 0.3          # Poisson's ratio (standard for NiTi/steel-like)
_G       = _E / (2 * (1 + _NU))
_R_BB    = 5e-4         # backbone radius [m]
_I_BB    = np.pi * _R_BB**4 / 4.0
_EIX     = _E * _I_BB   # bending stiffness [N·m²]
_EIY     = _EIX
_GJ      = _G * 2 * _I_BB  # torsional stiffness [N·m²]

# Tendon routing: 4 cables equidistant at r=8 mm from backbone axis
_R_TENDON   = 0.008     # tendon offset [m]
_R_LIST = [
    np.array([_R_TENDON,  0.0,         0.0]),
    np.array([0.0,         _R_TENDON,  0.0]),
    np.array([-_R_TENDON, 0.0,         0.0]),
    np.array([0.0,        -_R_TENDON,  0.0]),
]

# Modal curvature orders matching EKF state [k0x, k1x, k0y, k1y, k0z]
_ORDER_X = 1
_ORDER_Y = 1
_ORDER_Z = 0

# MAP prior standard deviations (matched to GT sampling range)
_SIGMA_FORCE_N   = 0.40   # N    (GT: U[-0.4, 0.4])
_SIGMA_MOMENT_NM = 0.04   # N·m  (GT: U[-0.04, 0.04])

# Additional cross-model uncertainty added to EKF-propagated Sigma_F
# so the MAP prior is not swamped by the (unrealistically small) EKF covariance.
_SIGMA_MODEL_FORCE_N   = 0.10   # N    (expected mismatch contribution)
_SIGMA_MODEL_MOMENT_NM = 0.010  # N·m

# Elastic energy Hessian (analytic, block-diagonal) — shared across calls
_Mx = gram_matrix(_ORDER_X)
_My = gram_matrix(_ORDER_Y)
_Mz = gram_matrix(_ORDER_Z)
_H_U: np.ndarray = _block_diag(
    (_EIX / _L) * _Mx,
    (_EIY / _L) * _My,
    (_GJ  / _L) * _Mz,
)


# ---------------------------------------------------------------------------
# Wrench estimation helpers
# ---------------------------------------------------------------------------

def _body_to_world_wrench(F_b: np.ndarray, R_tip: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 6D body wrench [moment_b; force_b] to world-frame (f, l)."""
    f_world = R_tip @ F_b[3:]
    l_world = R_tip @ F_b[:3]
    return f_world, l_world


def estimate_wrench_direct(
    m_est: np.ndarray,
    tau: np.ndarray,
    gamma: int = GAMMA_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Direct algebraic tip-wrench estimate (minimum-norm pseudo-inverse).

    Returns (f_world, l_world, F_body).
    """
    gradU  = elastic_energy_gradient(m_est, _EIX, _EIY, _GJ, _L,
                                     _ORDER_X, _ORDER_Y, _ORDER_Z)
    J_qm   = pull_jacobian(m_est, _R_LIST, _L, _ORDER_X, _ORDER_Y, _ORDER_Z)
    J_vbm, T_tip = body_jacobian_at_s(m_est, 1.0, gamma, _L,
                                      _ORDER_X, _ORDER_Y, _ORDER_Z)
    F_b   = solve_wrench(J_vbm, J_qm, gradU, tau)
    f_w, l_w = _body_to_world_wrench(F_b, T_tip[:3, :3])
    return f_w, l_w, F_b


def estimate_wrench_map(
    m_est: np.ndarray,
    tau: np.ndarray,
    P_est: np.ndarray,
    gamma: int = GAMMA_DEFAULT,
    sigma_force: float = _SIGMA_FORCE_N,
    sigma_moment: float = _SIGMA_MOMENT_NM,
    sigma_model_force: float = _SIGMA_MODEL_FORCE_N,
    sigma_model_moment: float = _SIGMA_MODEL_MOMENT_NM,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MAP-refined tip-wrench estimate.

    F̂ = argmin ||F - F̄||²_{Σ_F^{-1}} + ||F||²_{Q_F^{-1}}

    where:
      F̄      — direct estimate (pseudo-inverse)
      Σ_F    — world-frame wrench covariance propagated from EKF covariance P_est
               via the elastic energy Hessian and pinv(A.T), plus a
               diagonal cross-model uncertainty (sigma_model) term
      Q_F    — zero-mean Gaussian prior covariance (GT wrench range)

    Closed-form solution:
      F̂ = (Σ_F^{-1} + Q_F^{-1})^{-1} Σ_F^{-1} F̄

    Returns (f_world, l_world, F_body_direct).
    The _body field stores the direct estimate body wrench for reference.
    """
    gradU  = elastic_energy_gradient(m_est, _EIX, _EIY, _GJ, _L,
                                     _ORDER_X, _ORDER_Y, _ORDER_Z)
    J_qm   = pull_jacobian(m_est, _R_LIST, _L, _ORDER_X, _ORDER_Y, _ORDER_Z)
    J_vbm, T_tip = body_jacobian_at_s(m_est, 1.0, gamma, _L,
                                      _ORDER_X, _ORDER_Y, _ORDER_Z)

    # Direct estimate (body frame)
    F_b   = solve_wrench(J_vbm, J_qm, gradU, tau)
    R_tip = T_tip[:3, :3]
    f_bar, l_bar = _body_to_world_wrench(F_b, R_tip)   # world-frame F̄

    # Sigma_F: EKF-propagated + cross-model uncertainty (world frame, 3×3 each)
    A_pinv     = np.linalg.pinv(J_vbm.T, rcond=1e-8)   # (6, 5)
    Sigma_bw   = _H_U @ P_est @ _H_U.T                  # (5, 5)
    Sigma_Fb   = A_pinv @ Sigma_bw @ A_pinv.T            # (6, 6)
    Sigma_f = R_tip @ Sigma_Fb[3:, 3:] @ R_tip.T + sigma_model_force**2  * np.eye(3)
    Sigma_l = R_tip @ Sigma_Fb[:3, :3] @ R_tip.T + sigma_model_moment**2 * np.eye(3)

    # Q_F: prior covariance (uniform wrench range → spherical Gaussian proxy)
    Q_f = sigma_force**2  * np.eye(3)
    Q_l = sigma_moment**2 * np.eye(3)

    # MAP fusion: F̂ = (Σ^{-1} + Q^{-1})^{-1} Σ^{-1} F̄
    Sigma_f_inv = np.linalg.inv(Sigma_f)
    Sigma_l_inv = np.linalg.inv(Sigma_l)
    Q_f_inv     = np.linalg.inv(Q_f)
    Q_l_inv     = np.linalg.inv(Q_l)

    f_map = np.linalg.solve(Sigma_f_inv + Q_f_inv, Sigma_f_inv @ f_bar)
    l_map = np.linalg.solve(Sigma_l_inv + Q_l_inv, Sigma_l_inv @ l_bar)

    return f_map, l_map, F_b


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))))


def wrench_metrics(
    f_est: np.ndarray,
    l_est: np.ndarray,
    f_gt: np.ndarray,
    l_gt: np.ndarray,
) -> Dict[str, float]:
    force_err   = float(np.linalg.norm(f_est - f_gt))
    moment_err  = float(np.linalg.norm(l_est - l_gt))
    force_norm  = float(np.linalg.norm(f_gt))
    moment_norm = float(np.linalg.norm(l_gt))
    return {
        "force_err_N":        force_err,
        "moment_err_Nm":      moment_err,
        "force_dir_err_deg":  _angle_deg(f_est, f_gt),
        "moment_dir_err_deg": _angle_deg(l_est, l_gt),
        "nrmse_force":        force_err  / (force_norm  + 1e-12),
        "nrmse_moment":       moment_err / (moment_norm + 1e-12),
        "force_gt_norm_N":    force_norm,
        "moment_gt_norm_Nm":  moment_norm,
    }


# ---------------------------------------------------------------------------
# Per-layout evaluation
# ---------------------------------------------------------------------------

def evaluate_layout_wrenches(
    positions_gt: np.ndarray,
    orientations_gt: np.ndarray,
    case_ids_gt: np.ndarray,
    tau_gt: np.ndarray,
    f_ext_gt: np.ndarray,
    l_ext_gt: np.ndarray,
    R_meas: np.ndarray,
    imu_pos_norm: np.ndarray,
    e3: np.ndarray,
    gamma: int,
    meas_std_deg: float,
    alpha: float,
    P0: np.ndarray,
    steps: int,
    layout_name: str,
) -> List[Dict]:
    """Run EKF + wrench estimation for all cases × noise realizations."""
    num_cases, num_noise_real, n_imu, _, _ = R_meas.shape
    results: List[Dict] = []
    t0 = time.perf_counter()

    for case_idx in range(num_cases):
        tau  = tau_gt[case_idx]
        f_gt = f_ext_gt[case_idx]
        l_gt = l_ext_gt[case_idx]
        cid  = int(case_ids_gt[case_idx])

        for noise_idx in range(num_noise_real):
            R_frame = [R_meas[case_idx, noise_idx, si] for si in range(n_imu)]

            m_est, P_est = run_ekf_on_frame(
                R_frame=R_frame,
                imu_pos_norm=imu_pos_norm,
                e3=e3,
                gamma=gamma,
                meas_std_deg=meas_std_deg,
                alpha=alpha,
                P0=P0,
                steps=steps,
            )

            # Direct estimate
            try:
                f_dir, l_dir, Fb_dir = estimate_wrench_direct(m_est, tau, gamma)
                metrics_dir = wrench_metrics(f_dir, l_dir, f_gt, l_gt)
                ok_dir = True
            except Exception:
                f_dir = l_dir = Fb_dir = np.zeros(3)
                metrics_dir = wrench_metrics(np.zeros(3), np.zeros(3), f_gt, l_gt)
                ok_dir = False

            # MAP estimate
            try:
                f_map, l_map, Fb_map = estimate_wrench_map(m_est, tau, P_est, gamma)
                metrics_map = wrench_metrics(f_map, l_map, f_gt, l_gt)
                ok_map = True
            except Exception:
                f_map = l_map = Fb_map = np.zeros(3)
                metrics_map = wrench_metrics(np.zeros(3), np.zeros(3), f_gt, l_gt)
                ok_map = False

            base = {
                "case_id":        cid,
                "layout_name":    layout_name,
                "num_imus":       n_imu,
                "noise_real":     noise_idx,
                "_m_est":         m_est,
                "_f_gt":          f_gt,
                "_l_gt":          l_gt,
                "_f_est_direct":  f_dir,
                "_l_est_direct":  l_dir,
                "_f_est_map":     f_map,
                "_l_est_map":     l_map,
                "_Fb_direct":     Fb_dir,
                "_Fb_map":        Fb_map,
            }
            for method, metrics, ok in [("direct", metrics_dir, ok_dir),
                                        ("map",    metrics_map, ok_map)]:
                row = {**base, "method": method, "valid": ok}
                row.update(metrics)   # keys: force_err_N, moment_err_Nm, etc.
                results.append(row)

        if (case_idx + 1) % 10 == 0 or (case_idx + 1) == num_cases:
            elapsed = time.perf_counter() - t0
            print(f"    [{layout_name}] {case_idx + 1:3d}/{num_cases}  {elapsed:.1f} s")

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_wrench_results(results: List[Dict]) -> Dict[str, Dict]:
    """Aggregate metrics per (layout_name, method) pair."""
    from collections import defaultdict
    buckets: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in results:
        buckets[(r["layout_name"], r["method"])].append(r)

    summary: Dict[str, Dict] = {}
    metric_keys = [
        "force_err_N", "moment_err_Nm",
        "force_dir_err_deg", "moment_dir_err_deg",
        "nrmse_force", "nrmse_moment",
    ]
    for (layout, method), rows in sorted(buckets.items()):
        key = f"{layout}|{method}"
        entry: Dict = {"layout_name": layout, "method": method, "n_runs": len(rows)}
        for mk in metric_keys:
            vals = [r[mk] for r in rows if mk in r]
            if vals:
                entry[f"{mk}_mean"] = float(np.mean(vals))
                entry[f"{mk}_std"]  = float(np.std(vals))
                entry[f"{mk}_med"]  = float(np.median(vals))
        summary[key] = entry
    return summary


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_wrench_csv(results: List[Dict], path: Path) -> None:
    skip = {k for k in results[0] if k.startswith("_")}
    scalar_keys = [k for k in results[0] if k not in skip]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in scalar_keys})
    print(f"  CSV  -> {path}")


def save_wrench_summary_csv(summary: Dict[str, Dict], path: Path) -> None:
    if not summary:
        return
    all_keys = list(next(iter(summary.values())).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for entry in summary.values():
            writer.writerow(entry)
    print(f"  CSV  -> {path}")


def save_wrench_json(
    results: List[Dict],
    summary: Dict[str, Dict],
    cfg: dict,
    path: Path,
) -> None:
    skip = {k for k in results[0] if k.startswith("_")}
    clean = [{k: v for k, v in r.items() if k not in skip} for r in results]
    blob = {"config": cfg, "summary": summary, "results": clean}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
    print(f"  JSON -> {path}")


def save_wrench_npz(results: List[Dict], layout_names: List[str], path: Path) -> None:
    arrays: Dict[str, np.ndarray] = {"layout_names": np.array(layout_names)}
    for layout in layout_names:
        for method in ("direct", "map"):
            rows = [r for r in results
                    if r["layout_name"] == layout and r["method"] == method]
            if not rows:
                continue
            tag = f"{layout}_{method}"
            arrays[f"case_ids_{tag}"]   = np.array([r["case_id"]   for r in rows])
            arrays[f"noise_real_{tag}"] = np.array([r["noise_real"] for r in rows])
            f_key = "_f_est_direct" if method == "direct" else "_f_est_map"
            l_key = "_l_est_direct" if method == "direct" else "_l_est_map"
            arrays[f"f_est_{tag}"]      = np.array([r[f_key] for r in rows])
            arrays[f"l_est_{tag}"]      = np.array([r[l_key] for r in rows])
            arrays[f"f_gt_{tag}"]       = np.array([r["_f_gt"] for r in rows])
            arrays[f"l_gt_{tag}"]       = np.array([r["_l_gt"] for r in rows])
    np.savez_compressed(path, **arrays)
    print(f"  NPZ  -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: Tip-wrench estimation evaluation (Direct vs MAP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt",       default="gt_data/kirchhoff_gt_dataset.npz")
    parser.add_argument("--imu2",     default="gt_data/kirchhoff_imu_2imu.npz")
    parser.add_argument("--imu3",     default="gt_data/kirchhoff_imu_3imu.npz")
    parser.add_argument("--save-dir", default="gt_data/results")
    parser.add_argument("--gamma",    type=int,   default=GAMMA_DEFAULT)
    parser.add_argument("--alpha",    type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--meas-std-deg", type=float, default=MEAS_STD_DEG_DEF)
    parser.add_argument("--steps",    type=int,   default=STEPS_DEFAULT)
    parser.add_argument("--sigma-force",  type=float, default=_SIGMA_FORCE_N,
                        help="MAP force prior std dev [N]")
    parser.add_argument("--sigma-moment", type=float, default=_SIGMA_MOMENT_NM,
                        help="MAP moment prior std dev [N·m]")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    def _resolve(p: str, must_exist: bool = True) -> Path:
        raw = Path(p)
        out = raw if raw.is_absolute() else (script_dir / raw).resolve()
        if must_exist and not out.exists():
            raise FileNotFoundError(out)
        return out

    gt_path   = _resolve(args.gt)
    imu2_path = _resolve(args.imu2, must_exist=False)
    imu3_path = _resolve(args.imu3, must_exist=False)
    save_dir  = _resolve(args.save_dir, must_exist=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Load GT ---
    print("Loading ground-truth dataset ...")
    positions_gt, orientations_gt, case_ids_gt, gt_meta = \
        load_ground_truth_dataset(gt_path)
    gt_data = np.load(gt_path, allow_pickle=True)
    tau_gt   = gt_data["tau"]       # (N, 4)
    f_ext_gt = gt_data["f_ext"]     # (N, 3)
    l_ext_gt = gt_data["l_ext"]     # (N, 3)
    num_cases, num_pts, _ = positions_gt.shape
    L_phys = float(gt_meta.get("length_m", _L))
    print(f"  {num_cases} cases, L = {L_phys} m")
    print(f"  ||f_ext|| range: {np.linalg.norm(f_ext_gt, axis=1).min():.3f} "
          f"– {np.linalg.norm(f_ext_gt, axis=1).max():.3f} N")
    print(f"  ||l_ext|| range: {np.linalg.norm(l_ext_gt, axis=1).min():.4f} "
          f"– {np.linalg.norm(l_ext_gt, axis=1).max():.4f} N·m")

    # --- Load IMU measurements ---
    imu_datasets: Dict[str, Dict] = {}
    for label, path in [("2-IMU", imu2_path), ("3-IMU", imu3_path)]:
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {label}")
            continue
        R_true, R_meas, imu_pos, imu_idx, imu_meta = load_imu_measurements(path)
        imu_actual_s = np.array(
            imu_meta.get("imu_actual_s", (imu_idx / (num_pts - 1)).tolist()),
            dtype=float,
        )
        imu_datasets[label] = {
            "R_meas": R_meas,
            "imu_pos": imu_actual_s,
        }
        print(f"  {label}: R_meas {R_meas.shape},  imu_pos = "
              + "{" + ", ".join(f"{v:.4f}" for v in imu_actual_s) + "}")

    if not imu_datasets:
        raise RuntimeError("No IMU measurement files found. Run Step 2 first.")

    P0 = np.diag(P0_DIAG)
    all_results: List[Dict] = []

    for layout_name, ds in imu_datasets.items():
        print(f"\nEvaluating {layout_name} wrench estimation ...")
        layout_results = evaluate_layout_wrenches(
            positions_gt=positions_gt,
            orientations_gt=orientations_gt,
            case_ids_gt=case_ids_gt,
            tau_gt=tau_gt,
            f_ext_gt=f_ext_gt,
            l_ext_gt=l_ext_gt,
            R_meas=ds["R_meas"],
            imu_pos_norm=ds["imu_pos"],
            e3=E3.copy(),
            gamma=args.gamma,
            meas_std_deg=args.meas_std_deg,
            alpha=args.alpha,
            P0=P0,
            steps=args.steps,
            layout_name=layout_name,
        )
        all_results.extend(layout_results)
        print(f"  {len(layout_results)} runs complete.")

    # --- Summary table ---
    summary = aggregate_wrench_results(all_results)
    layout_names = list(imu_datasets.keys())

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Wrench Estimation Summary  (alpha={args.alpha}, steps={args.steps})")
    print(sep)
    hdr = f"{'Layout':<10}  {'Method':<8}  {'Force [N]':>14}  {'Moment [N·m]':>15}  {'F-dir [°]':>10}  {'M-dir [°]':>10}"
    print(hdr)
    print("-" * 80)
    for key, s in summary.items():
        print(
            f"{s['layout_name']:<10}  {s['method']:<8}  "
            f"{s['force_err_N_mean']*1e3:>8.2f}±{s['force_err_N_std']*1e3:<5.2f} mN  "
            f"{s['moment_err_Nm_mean']*1e3:>8.3f}±{s['moment_err_Nm_std']*1e3:<5.3f} mN·m  "
            f"{s['force_dir_err_deg_mean']:>8.2f}°  "
            f"{s['moment_dir_err_deg_mean']:>8.2f}°"
        )
    print(sep)

    # --- Save ---
    print("\nSaving results ...")
    save_wrench_csv(all_results, save_dir / "kirchhoff_wrench_est_results.csv")
    save_wrench_summary_csv(summary, save_dir / "kirchhoff_wrench_est_summary.csv")
    save_wrench_json(
        all_results, summary,
        cfg={
            "workflow": "Step-4 cross-model wrench estimation (Direct vs MAP)",
            "gt":            str(gt_path),
            "imu2":          str(imu2_path),
            "imu3":          str(imu3_path),
            "gamma":         args.gamma,
            "alpha":         args.alpha,
            "meas_std_deg":  args.meas_std_deg,
            "steps":         args.steps,
            "sigma_force_N": args.sigma_force,
            "sigma_moment_Nm": args.sigma_moment,
            "EIx_Nm2":      _EIX,
            "EIy_Nm2":      _EIY,
            "GJ_Nm2":       _GJ,
            "L_phys_m":     L_phys,
            "r_tendon_m":   _R_TENDON,
        },
        path=save_dir / "kirchhoff_wrench_est_results.json",
    )
    save_wrench_npz(all_results, layout_names, save_dir / "kirchhoff_wrench_est_data.npz")
    print("\nDone.")


if __name__ == "__main__":
    main()
