#!/usr/bin/env python
"""
EKF convergence/tuning sweep for sparse-IMU shape and wrench estimation.

This diagnostic varies repeated EKF update steps, measurement covariance scale,
initial covariance scale, process covariance scale, and initialization mode. It
does not change the virtual-work convention or wrench solvers.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from compare_oracle_vs_ekf_wrench_estimation import (
    MODAL_ORDERS,
    ModalOrder,
    estimate_modal_oracle,
    fit_sparse_imu_batch,
    fwd_transform_general,
    load_ground_truth_dataset,
    load_imu_measurements,
    wrench_from_modal,
    wrench_metrics,
)
from evaluate_kirchhoff_shape_estimation import (
    ALPHA_DEFAULT,
    E3,
    GAMMA_DEFAULT,
    MEAS_STD_DEG_DEF,
    P0_DIAG,
    Q_DIAG,
    so3_analytic_H_and_r,
    so3_log,
)
from virtual_work import body_jacobian_at_s
from wrench_tip_constrained_solver import (
    compute_b_w,
    make_S_direction,
    solve_load_subspace_wrench,
    world_wrench_from_body,
)


L_DEFAULT = 0.1
EPS_FORCE = 1e-12
EPS_RESIDUAL = 1e-12
ORDER_BY_LABEL = {o.label: o for o in MODAL_ORDERS}
METRIC_KEYS = [
    "gt_force_norm_N",
    "gt_moment_norm_Nm",
    "force_err_N",
    "moment_err_Nm",
    "nrmse_force",
    "nrmse_moment",
    "force_dir_err_deg",
    "moment_dir_err_deg",
    "known_dir_force_err_N",
    "known_dir_nrmse_force",
    "known_dir_force_dir_err_deg",
    "known_dir_residual_norm",
    "known_dir_residual_rel",
    "direct_residual_norm",
    "direct_residual_rel",
    "modal_rmse",
    "modal_err_norm",
    "tip_pos_err_m",
    "tip_rot_err_deg",
    "final_residual_norm",
    "final_NIS",
    "runtime_s",
]


def _parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("list must not be empty")
    if any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("all values must be > 0")
    return vals


def _parse_float_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("list must not be empty")
    return vals


def _resolve_path(script_dir: Path, raw_path: str, must_exist: bool = True) -> Path:
    expanded = Path(os.path.expandvars(raw_path))
    out = expanded if expanded.is_absolute() else (script_dir / expanded).resolve()
    if must_exist and not out.exists():
        raise FileNotFoundError(out)
    return out


def _n_params(order: ModalOrder) -> int:
    return (order.order_x + 1) + (order.order_y + 1) + (order.order_z + 1)


def _scaled_diag(base: Iterable[float], n: int, fill: float, scale: float) -> np.ndarray:
    vals = list(base)
    while len(vals) < n:
        vals.append(fill)
    return scale * np.diag(vals[:n])


def _safe_nanmean(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=float)
    return float("nan") if arr.size == 0 or np.all(np.isnan(arr)) else float(np.nanmean(arr))


def _safe_nanstd(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=float)
    return float("nan") if arr.size == 0 or np.all(np.isnan(arr)) else float(np.nanstd(arr))


def _safe_nanmedian(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=float)
    return float("nan") if arr.size == 0 or np.all(np.isnan(arr)) else float(np.nanmedian(arr))


def _rotation_residual_general(
    m: np.ndarray,
    imu_pos_norm: np.ndarray,
    R_frame: List[np.ndarray],
    order: ModalOrder,
    gamma: int,
    L_phys: float,
) -> np.ndarray:
    rows = []
    for s_i, R_meas in zip(imu_pos_norm, R_frame):
        R_pred = fwd_transform_general(
            m,
            float(s_i),
            gamma,
            L_phys,
            order.order_x,
            order.order_y,
            order.order_z,
        )[:3, :3]
        rows.append(so3_log(R_meas @ R_pred.T))
    return np.hstack(rows)


def _finite_difference_H(
    m: np.ndarray,
    imu_pos_norm: np.ndarray,
    R_frame: List[np.ndarray],
    order: ModalOrder,
    gamma: int,
    L_phys: float,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    r0 = _rotation_residual_general(m, imu_pos_norm, R_frame, order, gamma, L_phys)
    H = np.zeros((len(r0), len(m)), dtype=float)
    for j in range(len(m)):
        dm = np.zeros_like(m)
        dm[j] = eps
        rp = _rotation_residual_general(m + dm, imu_pos_norm, R_frame, order, gamma, L_phys)
        rm = _rotation_residual_general(m - dm, imu_pos_norm, R_frame, order, gamma, L_phys)
        H[:, j] = (rp - rm) / (2.0 * eps)
    return H, r0


def _H_and_r(
    m: np.ndarray,
    imu_pos_norm: np.ndarray,
    R_frame: List[np.ndarray],
    order: ModalOrder,
    gamma: int,
    L_phys: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if order.label == "(1,1,0)":
        return so3_analytic_H_and_r(m, imu_pos_norm, R_frame, E3.copy(), gamma)
    return _finite_difference_H(m, imu_pos_norm, R_frame, order, gamma, L_phys)


def run_tuned_ekf(
    R_frame: List[np.ndarray],
    imu_pos_norm: np.ndarray,
    order: ModalOrder,
    gamma: int,
    L_phys: float,
    meas_std_deg: float,
    alpha: float,
    p0_scale: float,
    q_scale: float,
    steps: int,
    init_m: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Run the same residual-level SO(3) EKF update with tunable P0/Q/init."""
    n = _n_params(order)
    P_est = _scaled_diag(P0_DIAG, n, float(P0_DIAG[-1]), p0_scale)
    Q = _scaled_diag(Q_DIAG, n, float(Q_DIAG[1]), q_scale)
    m_est = np.zeros(n, dtype=float) if init_m is None else np.asarray(init_m, dtype=float).copy()

    sigma = np.deg2rad(meas_std_deg)
    R_sngl = alpha * sigma**2 * np.eye(3)
    R_big = np.kron(np.eye(len(imu_pos_norm)), R_sngl)

    for _ in range(steps):
        P_pred = P_est + Q
        H, r = _H_and_r(m_est, imu_pos_norm, R_frame, order, gamma, L_phys)
        S = H @ P_pred @ H.T + R_big
        K = P_pred @ H.T @ np.linalg.pinv(S)
        m_est = m_est + K @ (-r)
        IKH = np.eye(n) - K @ H
        P_est = IKH @ P_pred @ IKH.T + K @ R_big @ K.T
        P_est = 0.5 * (P_est + P_est.T)

    Hf, rf = _H_and_r(m_est, imu_pos_norm, R_frame, order, gamma, L_phys)
    Sf = Hf @ P_est @ Hf.T + R_big
    final_residual_norm = float(np.linalg.norm(rf))
    final_nis = float(rf @ np.linalg.pinv(Sf) @ rf)
    return m_est, P_est, final_residual_norm, final_nis


def _tip_shape_metrics(
    m_est: np.ndarray,
    m_ref: np.ndarray | None,
    positions_case: np.ndarray,
    orientations_case: np.ndarray,
    order: ModalOrder,
    gamma: int,
    L_phys: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if m_ref is not None:
        err = m_est - m_ref
        out["modal_err_norm"] = float(np.linalg.norm(err))
        out["modal_rmse"] = float(np.sqrt(np.mean(err**2)))
    else:
        out["modal_err_norm"] = float("nan")
        out["modal_rmse"] = float("nan")

    T_tip = fwd_transform_general(
        m_est,
        1.0,
        gamma,
        L_phys,
        order.order_x,
        order.order_y,
        order.order_z,
    )
    p_gt = positions_case[-1]
    R_gt = orientations_case[-1].reshape(3, 3)
    out["tip_pos_err_m"] = float(np.linalg.norm(T_tip[:3, 3] - p_gt))
    out["tip_rot_err_deg"] = float(np.rad2deg(np.linalg.norm(so3_log(T_tip[:3, :3] @ R_gt.T))))
    return out


def _known_direction_metrics(
    m_est: np.ndarray,
    tau: np.ndarray,
    f_gt: np.ndarray,
    order: ModalOrder,
    gamma: int,
    L_phys: float,
) -> Dict[str, float]:
    if np.linalg.norm(f_gt) < EPS_FORCE:
        return {
            "known_dir_force_err_N": float("nan"),
            "known_dir_nrmse_force": float("nan"),
            "known_dir_force_dir_err_deg": float("nan"),
            "known_dir_residual_norm": float("nan"),
            "known_dir_residual_rel": float("nan"),
        }
    T_tip = fwd_transform_general(
        m_est,
        1.0,
        gamma,
        L_phys,
        order.order_x,
        order.order_y,
        order.order_z,
    )
    d_world = f_gt / np.linalg.norm(f_gt)
    d_body = T_tip[:3, :3].T @ d_world
    S_dir = make_S_direction(d_body)
    w_body, z_hat, J_vbm, T_solver, b_w = solve_load_subspace_wrench(
        m_est,
        tau,
        S_dir,
        gamma=gamma,
        order_x=order.order_x,
        order_y=order.order_y,
        order_z=order.order_z,
        L=L_phys,
    )
    residual = (J_vbm.T @ S_dir) @ z_hat - b_w
    f_world, _ = world_wrench_from_body(w_body, T_solver)
    force_err = float(np.linalg.norm(f_world - f_gt))
    return {
        "known_dir_force_err_N": force_err,
        "known_dir_nrmse_force": force_err / (float(np.linalg.norm(f_gt)) + 1e-12),
        "known_dir_force_dir_err_deg": _angle_deg(f_world, f_gt),
        "known_dir_residual_norm": float(np.linalg.norm(residual)),
        "known_dir_residual_rel": float(
            np.linalg.norm(residual) / max(float(np.linalg.norm(b_w)), EPS_RESIDUAL)
        ),
    }


def _empty_known_direction_metrics(
    applicable: bool,
    skip_reason: str,
    warning: str = "",
) -> Dict[str, float | bool | str]:
    return {
        "known_dir_force_err_N": float("nan"),
        "known_dir_nrmse_force": float("nan"),
        "known_dir_force_dir_err_deg": float("nan"),
        "known_dir_residual_norm": float("nan"),
        "known_dir_residual_rel": float("nan"),
        "known_dir_applicable": bool(applicable),
        "known_dir_skip_reason": skip_reason,
        "known_dir_warning": warning,
    }


def _direct_residual_metrics(
    m_est: np.ndarray,
    tau: np.ndarray,
    order: ModalOrder,
    gamma: int,
    L_phys: float,
) -> Dict[str, float]:
    b_w = compute_b_w(
        m_est,
        tau,
        order_x=order.order_x,
        order_y=order.order_y,
        order_z=order.order_z,
        L=L_phys,
    )
    J_vbm, _ = body_jacobian_at_s(
        m_est,
        1.0,
        gamma,
        L_phys,
        order.order_x,
        order.order_y,
        order.order_z,
    )
    A = J_vbm.T
    w_body = np.linalg.pinv(A, rcond=1e-8) @ b_w
    residual = A @ w_body - b_w
    return {
        "direct_residual_norm": float(np.linalg.norm(residual)),
        "direct_residual_rel": float(
            np.linalg.norm(residual) / max(float(np.linalg.norm(b_w)), EPS_RESIDUAL)
        ),
    }


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.rad2deg(np.arccos(np.clip(float(a @ b) / (na * nb), -1.0, 1.0))))


def _init_modal_state(
    init_mode: str,
    oracle_m: np.ndarray | None,
    R_frame: List[np.ndarray],
    imu_pos_norm: np.ndarray,
    order: ModalOrder,
    gamma: int,
    L_phys: float,
    alpha: float,
    meas_std_deg: float,
    batch_max_iter: int,
    batch_multistart: int,
    seed: int,
) -> Tuple[np.ndarray | None, str, Dict[str, float | str | bool]]:
    if init_mode == "zero":
        return None, "zero", {}
    if init_mode == "oracle":
        if oracle_m is not None:
            return oracle_m, "oracle", {}
        return None, "zero_fallback_oracle_unavailable", {}
    if init_mode == "sparse_batch":
        try:
            fit = fit_sparse_imu_batch(
                R_meas=np.asarray(R_frame),
                imu_positions=imu_pos_norm,
                order_x=order.order_x,
                order_y=order.order_y,
                order_z=order.order_z,
                L=L_phys,
                gamma=gamma,
                sigma_rot_rad=np.sqrt(alpha) * np.deg2rad(meas_std_deg),
                max_iter=batch_max_iter,
                multistart=batch_multistart,
                seed=seed,
            )
            m_batch = fit[0]
            success = bool(fit[2])
            extra = {
                "init_batch_cost": float(fit[1]),
                "init_batch_success": success,
                "init_batch_nfev": int(fit[3]),
                "init_batch_residual_norm": float(fit[4]),
                "init_batch_solver": str(fit[5]),
            }
            if len(fit) >= 9:
                extra["init_batch_status"] = int(fit[6])
                extra["init_batch_message_short"] = str(fit[7])
                extra["init_batch_hit_limit"] = bool(fit[8])
            if success:
                return m_batch, "sparse_batch", extra
            return None, "zero_fallback_sparse_batch_failed", extra
        except Exception as exc:
            return None, "zero_fallback_sparse_batch_exception", {
                "init_batch_error": str(exc)[:160]
            }
    raise ValueError(f"Unknown init mode: {init_mode}")


def _save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV  -> {path}")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _save_json(config: Dict, results: List[Dict], summary: List[Dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"config": config, "results": results, "summary": summary},
            f,
            indent=2,
            default=_json_default,
        )
    print(f"  JSON -> {path}")


def _aggregate(rows: List[Dict]) -> List[Dict]:
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["modal_order"],
            row["layout_name"],
            row["steps"],
            row["alpha"],
            row["p0_scale"],
            row["q_scale"],
            row["init_mode"],
        )
        groups[key].append(row)

    summary: List[Dict] = []
    for key, items in sorted(groups.items()):
        modal_order, layout_name, steps, alpha, p0_scale, q_scale, init_mode = key
        entry: Dict = {
            "modal_order": modal_order,
            "layout_name": layout_name,
            "steps": steps,
            "alpha": alpha,
            "p0_scale": p0_scale,
            "q_scale": q_scale,
            "init_mode": init_mode,
            "n_runs": len(items),
        }
        for metric in METRIC_KEYS:
            vals = [row.get(metric, float("nan")) for row in items]
            entry[f"{metric}_mean"] = _safe_nanmean(vals)
            entry[f"{metric}_std"] = _safe_nanstd(vals)
            entry[f"{metric}_med"] = _safe_nanmedian(vals)
        applicable_vals = [
            float(row["known_dir_applicable"])
            for row in items
            if isinstance(row.get("known_dir_applicable"), (bool, np.bool_))
        ]
        entry["known_dir_applicable_rate"] = (
            float("nan") if not applicable_vals else float(np.mean(applicable_vals))
        )
        skip_counts: Dict[str, int] = defaultdict(int)
        for row in items:
            reason = str(row.get("known_dir_skip_reason", ""))
            if reason:
                skip_counts[reason] += 1
        entry["known_dir_skip_reason_counts"] = ";".join(
            f"{reason}:{count}" for reason, count in sorted(skip_counts.items())
        )
        summary.append(entry)
    return summary


def _write_markdown(summary: List[Dict], config: Dict, path: Path) -> None:
    lines = [
        "# EKF Convergence Sweep Summary",
        "",
        "This diagnostic varies EKF update steps, measurement covariance scaling,",
        "initial covariance scaling, process covariance scaling, and initialization.",
        "",
        "Interpretation guide:",
        "- If error decreases with steps, convergence/local-linearization matters.",
        "- If oracle or sparse_batch initialization greatly improves EKF, initialization matters.",
        "- If error remains high even with many steps and good initialization, sparse sensing or wrench conditioning dominates.",
        "- If known-direction 1D is much better than direct 6D, load-subspace conditioning dominates.",
        "",
        "Known-direction 1D note:",
        "The known-direction 1D estimator assumes a force-only wrench. It is reported only for cases with negligible GT moment by default. On general force+moment datasets, use --known-dir-allow-nonzero-moment only as a diagnostic; large errors indicate model mismatch rather than failure of the known-direction estimator.",
        "",
        "Configuration:",
    ]
    for key, value in config.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "| modal_order | layout | steps | alpha | p0_scale | q_scale | init | n | nrmse_F_mean | nrmse_M_mean | known_dir_applicable_rate | known_dir_nrmse_F_mean | known_dir_resid_rel_mean | direct_resid_rel_mean |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary[:40]:
        lines.append(
            "| {modal_order} | {layout_name} | {steps} | {alpha:g} | {p0_scale:g} | {q_scale:g} | "
            "{init_mode} | {n_runs} | {nf:.4g} | {nm:.4g} | {kda:.4g} | {kdf:.4g} | {kdr:.4g} | {dr:.4g} |".format(
                nf=row.get("nrmse_force_mean", float("nan")),
                nm=row.get("nrmse_moment_mean", float("nan")),
                kda=row.get("known_dir_applicable_rate", float("nan")),
                kdf=row.get("known_dir_nrmse_force_mean", float("nan")),
                kdr=row.get("known_dir_residual_rel_mean", float("nan")),
                dr=row.get("direct_residual_rel_mean", float("nan")),
                **row,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  MD   -> {path}")


def run_sweep(args: argparse.Namespace) -> Tuple[List[Dict], List[Dict], Dict]:
    script_dir = Path(__file__).resolve().parent
    gt_path = _resolve_path(script_dir, args.gt)
    imu2_path = _resolve_path(script_dir, args.imu2, must_exist=False)
    imu3_path = _resolve_path(script_dir, args.imu3, must_exist=False)

    save_dir = _resolve_path(script_dir, args.save_dir, must_exist=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets ...")
    positions_gt, orientations_gt, case_ids_gt, gt_meta = load_ground_truth_dataset(gt_path)
    gt_data = np.load(gt_path, allow_pickle=True)
    tau_gt = gt_data["tau"]
    f_ext_gt = gt_data["f_ext"]
    l_ext_gt = gt_data["l_ext"]
    L_phys = float(gt_meta.get("length_m", L_DEFAULT))

    n_cases = min(args.max_cases, len(positions_gt))
    print(f"Applying --max-cases {args.max_cases}: using first {n_cases} cases.")
    positions_gt = positions_gt[:n_cases]
    orientations_gt = orientations_gt[:n_cases]
    case_ids_gt = case_ids_gt[:n_cases]
    tau_gt = tau_gt[:n_cases]
    f_ext_gt = f_ext_gt[:n_cases]
    l_ext_gt = l_ext_gt[:n_cases]

    selected_layouts = ["2-IMU", "3-IMU"] if args.layouts == "both" else [args.layouts]
    imu_datasets: Dict[str, Dict] = {}
    for label, path in [("2-IMU", imu2_path), ("3-IMU", imu3_path)]:
        if label not in selected_layouts:
            continue
        if not path.exists():
            print(f"  WARNING: {path} missing; skipping {label}")
            continue
        _, R_meas, _, imu_idx, imu_meta = load_imu_measurements(path)
        R_meas = R_meas[:n_cases]
        n_noise = min(args.n_noise, R_meas.shape[1])
        print(f"Applying --n-noise {args.n_noise} for {label}: using first {n_noise} noise realizations.")
        R_meas = R_meas[:, :n_noise]
        num_pts = positions_gt.shape[1]
        imu_actual_s = np.array(
            imu_meta.get("imu_actual_s", (imu_idx / (num_pts - 1)).tolist()),
            dtype=float,
        )
        imu_datasets[label] = {"R_meas": R_meas, "imu_pos": imu_actual_s}
        print(f"  {label}: R_meas {R_meas.shape}, imu_pos={imu_actual_s}")
    if not imu_datasets:
        raise RuntimeError("No IMU measurement datasets selected/found.")

    orders = [ORDER_BY_LABEL[label] for label in args.orders]
    oracle_cache: Dict[Tuple[str, int], np.ndarray | None] = {}
    for order in orders:
        for ci in range(n_cases):
            try:
                oracle_cache[(order.label, ci)] = estimate_modal_oracle(
                    positions_gt[ci],
                    orientations_gt[ci],
                    order.order_x,
                    order.order_y,
                    order.order_z,
                    L_phys,
                )
            except Exception:
                oracle_cache[(order.label, ci)] = None

    rows: List[Dict] = []
    for order in orders:
        n_m = _n_params(order)
        for layout_name, ds in imu_datasets.items():
            R_meas = ds["R_meas"]
            imu_pos = ds["imu_pos"]
            for steps in args.steps_list:
                for alpha in args.alpha_list:
                    for p0_scale in args.p0_scale_list:
                        for q_scale in args.q_scale_list:
                            print(
                                f"Running {order.label} {layout_name}: steps={steps}, "
                                f"alpha={alpha}, p0_scale={p0_scale}, q_scale={q_scale}, "
                                f"init={args.init_mode}"
                            )
                            for ci in range(n_cases):
                                oracle_m = oracle_cache[(order.label, ci)]
                                for ni in range(R_meas.shape[1]):
                                    R_frame = [R_meas[ci, ni, si] for si in range(len(imu_pos))]
                                    seed_i = args.seed + 10000 * ci + 100 * ni + steps
                                    init_m, init_used, init_extra = _init_modal_state(
                                        args.init_mode,
                                        oracle_m,
                                        R_frame,
                                        imu_pos,
                                        order,
                                        args.gamma,
                                        L_phys,
                                        alpha,
                                        args.meas_std_deg,
                                        args.batch_max_iter,
                                        args.batch_multistart,
                                        seed_i,
                                    )
                                    t0 = time.perf_counter()
                                    try:
                                        m_est, P_est, residual_norm, nis = run_tuned_ekf(
                                            R_frame,
                                            imu_pos,
                                            order,
                                            args.gamma,
                                            L_phys,
                                            args.meas_std_deg,
                                            alpha,
                                            p0_scale,
                                            q_scale,
                                            steps,
                                            init_m=init_m,
                                        )
                                        runtime_s = time.perf_counter() - t0
                                        f_est, l_est = wrench_from_modal(
                                            m_est,
                                            tau_gt[ci],
                                            order.order_x,
                                            order.order_y,
                                            order.order_z,
                                            args.gamma,
                                        )
                                        metrics = wrench_metrics(f_est, l_est, f_ext_gt[ci], l_ext_gt[ci])
                                        metrics.update(
                                            _direct_residual_metrics(
                                                m_est,
                                                tau_gt[ci],
                                                order,
                                                args.gamma,
                                                L_phys,
                                            )
                                        )
                                        valid = True
                                    except Exception as exc:
                                        runtime_s = time.perf_counter() - t0
                                        m_est = np.zeros(n_m)
                                        P_est = np.full((n_m, n_m), np.nan)
                                        residual_norm = float("nan")
                                        nis = float("nan")
                                        metrics = wrench_metrics(
                                            np.zeros(3),
                                            np.zeros(3),
                                            f_ext_gt[ci],
                                            l_ext_gt[ci],
                                        )
                                        metrics.update(
                                            {
                                                "direct_residual_norm": float("nan"),
                                                "direct_residual_rel": float("nan"),
                                            }
                                        )
                                        valid = False
                                        init_extra["run_error"] = str(exc)[:160]

                                    gt_force_norm = float(np.linalg.norm(f_ext_gt[ci]))
                                    gt_moment_norm = float(np.linalg.norm(l_ext_gt[ci]))
                                    metrics["gt_force_norm_N"] = gt_force_norm
                                    metrics["gt_moment_norm_Nm"] = gt_moment_norm
                                    metrics.update(
                                        _tip_shape_metrics(
                                            m_est,
                                            oracle_m,
                                            positions_gt[ci],
                                            orientations_gt[ci],
                                            order,
                                            args.gamma,
                                            L_phys,
                                        )
                                    )
                                    known_dir_applicable = (
                                        gt_force_norm > EPS_FORCE
                                        and gt_moment_norm <= args.known_dir_moment_threshold
                                    )
                                    if not args.enable_known_direction_1d:
                                        metrics.update(
                                            _empty_known_direction_metrics(
                                                applicable=False,
                                                skip_reason="disabled",
                                            )
                                        )
                                    elif not valid:
                                        metrics.update(
                                            _empty_known_direction_metrics(
                                                applicable=False,
                                                skip_reason="invalid_ekf_run",
                                            )
                                        )
                                    elif gt_force_norm <= EPS_FORCE:
                                        metrics.update(
                                            _empty_known_direction_metrics(
                                                applicable=False,
                                                skip_reason="zero_gt_force",
                                            )
                                        )
                                    elif known_dir_applicable:
                                        kd_metrics = _known_direction_metrics(
                                            m_est,
                                            tau_gt[ci],
                                            f_ext_gt[ci],
                                            order,
                                            args.gamma,
                                            L_phys,
                                        )
                                        kd_metrics["known_dir_applicable"] = True
                                        kd_metrics["known_dir_skip_reason"] = ""
                                        kd_metrics["known_dir_warning"] = ""
                                        metrics.update(kd_metrics)
                                    elif args.known_dir_allow_nonzero_moment:
                                        kd_metrics = _known_direction_metrics(
                                            m_est,
                                            tau_gt[ci],
                                            f_ext_gt[ci],
                                            order,
                                            args.gamma,
                                            L_phys,
                                        )
                                        kd_metrics["known_dir_applicable"] = False
                                        kd_metrics["known_dir_skip_reason"] = (
                                            "forced_diagnostic_nonzero_gt_moment"
                                        )
                                        kd_metrics["known_dir_warning"] = (
                                            "nonzero GT moment; force-only known-direction model is diagnostic only"
                                        )
                                        metrics.update(kd_metrics)
                                    else:
                                        metrics.update(
                                            _empty_known_direction_metrics(
                                                applicable=False,
                                                skip_reason="nonzero_gt_moment",
                                            )
                                        )

                                    row = {
                                        "modal_order": order.label,
                                        "order_x": order.order_x,
                                        "order_y": order.order_y,
                                        "order_z": order.order_z,
                                        "n_m": n_m,
                                        "layout_name": layout_name,
                                        "steps": steps,
                                        "alpha": alpha,
                                        "p0_scale": p0_scale,
                                        "q_scale": q_scale,
                                        "init_mode": args.init_mode,
                                        "init_used": init_used,
                                        "case_idx": ci,
                                        "case_id": int(case_ids_gt[ci]),
                                        "noise_idx": ni,
                                        "valid": valid,
                                        "final_residual_norm": residual_norm,
                                        "final_NIS": nis,
                                        "runtime_s": runtime_s,
                                    }
                                    row.update(init_extra)
                                    row.update(metrics)
                                    rows.append(row)

    summary = _aggregate(rows)
    config = {
        "gt": str(gt_path),
        "imu2": str(imu2_path),
        "imu3": str(imu3_path),
        "save_dir": str(save_dir),
        "orders": args.orders,
        "max_cases": args.max_cases,
        "n_noise": args.n_noise,
        "layouts": args.layouts,
        "steps_list": args.steps_list,
        "alpha_list": args.alpha_list,
        "p0_scale_list": args.p0_scale_list,
        "q_scale_list": args.q_scale_list,
        "init_mode": args.init_mode,
        "enable_known_direction_1d": args.enable_known_direction_1d,
        "known_dir_moment_threshold": args.known_dir_moment_threshold,
        "known_dir_allow_nonzero_moment": args.known_dir_allow_nonzero_moment,
        "gamma": args.gamma,
        "meas_std_deg": args.meas_std_deg,
        "L_phys_m": L_phys,
        "seed": args.seed,
        "batch_max_iter": args.batch_max_iter,
        "batch_multistart": args.batch_multistart,
    }

    _save_csv(rows, save_dir / "ekf_convergence_sweep_results.csv")
    _save_csv(summary, save_dir / "ekf_convergence_sweep_summary.csv")
    _save_json(config, rows, summary, save_dir / "ekf_convergence_sweep_results.json")
    _write_markdown(summary, config, save_dir / "ekf_convergence_sweep_summary.md")

    print("\nSummary preview:")
    for row in summary[:12]:
        print(
            f"  {row['modal_order']} {row['layout_name']} steps={row['steps']} "
            f"init={row['init_mode']} nrmse_F={row['nrmse_force_mean']:.4g} "
            f"nrmse_M={row['nrmse_moment_mean']:.4g} "
            f"known_dir_applicable={row['known_dir_applicable_rate']:.4g} "
            f"known_dir_F={row['known_dir_nrmse_force_mean']:.4g}"
        )
    return rows, summary, config


def build_parser() -> argparse.ArgumentParser:
    default_save = str(
        Path(tempfile.gettempdir())
        / "shape-force-est-IMU-smoke"
        / "ekf_convergence_sweep"
    )
    parser = argparse.ArgumentParser(
        description="Sweep EKF convergence/tuning for sparse-IMU wrench estimation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt", default="gt_data/kirchhoff_gt_dataset.npz")
    parser.add_argument("--imu2", default="gt_data/kirchhoff_imu_2imu.npz")
    parser.add_argument("--imu3", default="gt_data/kirchhoff_imu_3imu.npz")
    parser.add_argument("--save-dir", default=default_save)
    parser.add_argument(
        "--orders",
        nargs="+",
        default=["(1,1,0)"],
        metavar="ORDER",
        help="modal order labels, e.g. \"(1,1,0)\" \"(1,2,0)\"",
    )
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--n-noise", type=int, default=1)
    parser.add_argument("--layouts", choices=["2-IMU", "3-IMU", "both"], default="both")
    parser.add_argument("--steps-list", type=_parse_int_list, default=_parse_int_list("1,5,10,20,50"))
    parser.add_argument("--alpha-list", type=_parse_float_list, default=_parse_float_list("1.0"))
    parser.add_argument("--p0-scale-list", type=_parse_float_list, default=_parse_float_list("1.0"))
    parser.add_argument("--q-scale-list", type=_parse_float_list, default=_parse_float_list("1.0"))
    parser.add_argument("--init-mode", choices=["zero", "oracle", "sparse_batch"], default="zero")
    parser.add_argument("--enable-known-direction-1d", action="store_true")
    parser.add_argument("--known-dir-moment-threshold", type=float, default=1e-9)
    parser.add_argument("--known-dir-allow-nonzero-moment", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-max-iter", type=int, default=50)
    parser.add_argument("--batch-multistart", type=int, default=1)
    parser.add_argument("--gamma", type=int, default=GAMMA_DEFAULT)
    parser.add_argument("--meas-std-deg", type=float, default=MEAS_STD_DEG_DEF)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_cases <= 0:
        parser.error("--max-cases must be > 0")
    if args.n_noise <= 0:
        parser.error("--n-noise must be > 0")
    if args.batch_max_iter <= 0:
        parser.error("--batch-max-iter must be > 0")
    if args.batch_multistart <= 0:
        parser.error("--batch-multistart must be > 0")
    if any(v < 0.0 for v in args.p0_scale_list):
        parser.error("--p0-scale-list values must be >= 0")
    if any(v < 0.0 for v in args.q_scale_list):
        parser.error("--q-scale-list values must be >= 0")
    if any(v <= 0.0 for v in args.alpha_list):
        parser.error("--alpha-list values must be > 0")
    if args.known_dir_moment_threshold < 0.0:
        parser.error("--known-dir-moment-threshold must be >= 0")
    bad_orders = [label for label in args.orders if label not in ORDER_BY_LABEL]
    if bad_orders:
        parser.error(
            "unknown --orders value(s): "
            + ", ".join(bad_orders)
            + ". Valid values: "
            + ", ".join(ORDER_BY_LABEL)
        )
    run_sweep(args)


if __name__ == "__main__":
    main()
