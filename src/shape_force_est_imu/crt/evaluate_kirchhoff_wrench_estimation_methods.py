#!/usr/bin/env python3
"""
Cross-model tip-wrench estimation follow-up study.

This new script keeps the original `evaluate_kirchhoff_wrench_estimation.py`
untouched and evaluates three estimator variants on the same static
cross-model dataset:

  1. direct
     Minimum-norm 6D reconstruction:
         F_bar = pinv(J_Vb_m.T) @ b_w

  2. recursive_map
     Corrected recursive MAP refinement:
         argmin ||F_k - F_bar,k||^2_{Sigma_F,k^{-1}}
              + ||F_k - F^-_k||^2_{Q_F^{-1}}
     with
         F^-_k = F_hat_{k-1}
     and initialization
         F_hat_1 = F_bar,1

  3. moment_only
     Observability-informed constrained estimator motivated by the new
     rank/singular-vector analysis. It solves only for body-tip moments and
     treats force recovery as not observable enough under the current 5D model.

Important interpretation note
-----------------------------
The current shape-estimation pipeline is still the original 5D EKF
`[k0x, k1x, k0y, k1y, k0z]`. This script changes only the downstream wrench
reconstruction stage. It does not re-derive a higher-order EKF.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    so3_analytic_H_and_r,
)
from wrench_study_utils import (
    BASELINE_ORDER,
    EIX_DEFAULT,
    EIY_DEFAULT,
    GJ_DEFAULT,
    L_DEFAULT,
    body_wrench_to_world,
    build_virtual_work_terms,
    direct_wrench_body_from_terms,
    moment_only_wrench_body_from_terms,
    propagate_wrench_covariance_body,
    recursive_map_update,
    wrench_metrics_from_world6,
)


METHOD_ORDER = ["direct", "recursive_map", "moment_only"]
METHOD_LABELS = {
    "direct": "Direct",
    "recursive_map": "Recursive MAP",
    "moment_only": "Moment-only",
}
METHOD_COLORS = {
    "direct": "#1f77b4",
    "recursive_map": "#d62728",
    "moment_only": "#2a9d8f",
}
LAYOUT_LINESTYLES = {
    "2-IMU": "--",
    "3-IMU": "-",
}


def resolve_cli_path(path_str: str, script_dir: Path, must_exist: bool) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        path = raw
    elif raw.exists():
        path = raw.resolve()
    else:
        path = (script_dir / raw).resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def run_ekf_history_on_frame(
    R_frame: List[np.ndarray],
    imu_pos_norm: np.ndarray,
    e3: np.ndarray,
    gamma: int,
    meas_std_deg: float,
    alpha: float,
    P0: np.ndarray,
    steps: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Same EKF as the current cross-model shape-estimation baseline, but returns
    the full step-by-step modal state and covariance history.
    """
    sigma = np.deg2rad(meas_std_deg)
    R_single = alpha * sigma**2 * np.eye(3)
    Q_modal = np.diag(Q_DIAG)
    n_imu = len(imu_pos_norm)
    R_big = np.kron(np.eye(n_imu), R_single)

    m_est = np.zeros(STATE_DIM)
    P_est = P0.copy()
    history: List[Tuple[np.ndarray, np.ndarray]] = []

    for _ in range(steps):
        P_pred = P_est + Q_modal
        H, r = so3_analytic_H_and_r(m_est, imu_pos_norm, R_frame, e3, gamma)
        innov = -r
        S = H @ P_pred @ H.T + R_big
        K = P_pred @ H.T @ np.linalg.inv(S)
        m_est = m_est + K @ innov
        IKH = np.eye(STATE_DIM) - K @ H
        P_est = IKH @ P_pred @ IKH.T + K @ R_big @ K.T
        P_est = 0.5 * (P_est + P_est.T)
        history.append((m_est.copy(), P_est.copy()))

    return history


def nan_summary(values: np.ndarray) -> Tuple[float, float, float]:
    if np.all(np.isnan(values)):
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmean(values)), float(np.nanstd(values)), float(np.nanmedian(values))


def aggregate_final_results(results: List[Dict]) -> Dict[str, Dict]:
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in results:
        groups[(row["layout_name"], row["method"])].append(row)

    summary: Dict[str, Dict] = {}
    metric_keys = [
        "force_err_N",
        "force_norm_error",
        "force_mag_rel_error",
        "force_dir_err_deg",
        "moment_err_Nm",
        "moment_norm_error",
        "moment_mag_rel_error",
        "moment_dir_err_deg",
    ]
    for (layout_name, method), rows in groups.items():
        entry = {
            "layout_name": layout_name,
            "method": method,
            "n_runs": len(rows),
            "force_metrics_applicable": bool(np.any([r["force_metrics_applicable"] for r in rows])),
            "moment_metrics_applicable": bool(np.any([r["moment_metrics_applicable"] for r in rows])),
        }
        for key in metric_keys:
            vals = np.array([r[key] for r in rows], dtype=float)
            mean, std, med = nan_summary(vals)
            entry[f"{key}_mean"] = mean
            entry[f"{key}_std"] = std
            entry[f"{key}_med"] = med
        summary[f"{layout_name}|{method}"] = entry
    return summary


def aggregate_stepwise(stepwise_rows: List[Dict], steps: int) -> Dict[str, Dict]:
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in stepwise_rows:
        groups[(row["layout_name"], row["method"])].append(row)

    summary: Dict[str, Dict] = {}
    for (layout_name, method), rows in groups.items():
        force_err = np.stack([r["force_err_traj"] for r in rows], axis=0)
        moment_err = np.stack([r["moment_err_traj"] for r in rows], axis=0)
        force_dir = np.stack([r["force_dir_traj"] for r in rows], axis=0)
        moment_dir = np.stack([r["moment_dir_traj"] for r in rows], axis=0)

        if np.all(np.isnan(force_err)):
            force_err_mean_over_steps = float("nan")
            force_dir_mean_over_steps = float("nan")
            force_err_traj_mean = np.full(force_err.shape[1], np.nan)
            force_err_traj_std = np.full(force_err.shape[1], np.nan)
            force_dir_traj_mean = np.full(force_dir.shape[1], np.nan)
        else:
            force_err_mean_over_steps = float(np.nanmean(force_err))
            force_dir_mean_over_steps = float(np.nanmean(force_dir))
            force_err_traj_mean = np.nanmean(force_err, axis=0)
            force_err_traj_std = np.nanstd(force_err, axis=0)
            force_dir_traj_mean = np.nanmean(force_dir, axis=0)

        summary[f"{layout_name}|{method}"] = {
            "layout_name": layout_name,
            "method": method,
            "n_runs": len(rows),
            "steps": steps,
            "force_err_mean_over_steps_N": force_err_mean_over_steps,
            "moment_err_mean_over_steps_Nm": float(np.nanmean(moment_err)),
            "force_dir_mean_over_steps_deg": force_dir_mean_over_steps,
            "moment_dir_mean_over_steps_deg": float(np.nanmean(moment_dir)),
            "force_err_traj_mean": force_err_traj_mean,
            "force_err_traj_std": force_err_traj_std,
            "moment_err_traj_mean": np.nanmean(moment_err, axis=0),
            "moment_err_traj_std": np.nanstd(moment_err, axis=0),
            "force_dir_traj_mean": force_dir_traj_mean,
            "moment_dir_traj_mean": np.nanmean(moment_dir, axis=0),
        }
    return summary


def save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV  -> {path}")


def save_summary_csv(summary: Dict[str, Dict], path: Path) -> None:
    if not summary:
        return
    rows = list(summary.values())
    fields = [
        key
        for key, value in rows[0].items()
        if not isinstance(value, np.ndarray)
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    print(f"  CSV  -> {path}")


def save_json(blob: dict, path: Path) -> None:
    def _convert(obj):
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if isinstance(obj, tuple):
            return [_convert(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    path.write_text(json.dumps(_convert(blob), indent=2), encoding="utf-8")
    print(f"  JSON -> {path}")


def save_stepwise_npz(
    stepwise_summary: Dict[str, Dict],
    path: Path,
) -> None:
    arrays: Dict[str, np.ndarray] = {}
    layout_names = sorted({row["layout_name"] for row in stepwise_summary.values()})
    arrays["layout_names"] = np.array(layout_names, dtype=object)
    arrays["method_names"] = np.array(METHOD_ORDER, dtype=object)
    for key, row in stepwise_summary.items():
        tag = f"{row['layout_name']}_{row['method']}"
        arrays[f"force_err_traj_mean_{tag}"] = row["force_err_traj_mean"]
        arrays[f"force_err_traj_std_{tag}"] = row["force_err_traj_std"]
        arrays[f"moment_err_traj_mean_{tag}"] = row["moment_err_traj_mean"]
        arrays[f"moment_err_traj_std_{tag}"] = row["moment_err_traj_std"]
        arrays[f"force_dir_traj_mean_{tag}"] = row["force_dir_traj_mean"]
        arrays[f"moment_dir_traj_mean_{tag}"] = row["moment_dir_traj_mean"]
    np.savez_compressed(path, **arrays)
    print(f"  NPZ  -> {path}")


def build_comparison_figure(
    final_summary: Dict[str, Dict],
    stepwise_summary: Dict[str, Dict],
    steps: int,
    save_stem: Path,
) -> None:
    layouts = ["2-IMU", "3-IMU"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_force_bar, ax_force_step = axes[0]
    ax_moment_bar, ax_moment_step = axes[1]

    # Final-step grouped bars
    x = np.arange(len(layouts))
    width = 0.24

    force_methods = ["direct", "recursive_map"]
    for idx, method in enumerate(force_methods):
        means = []
        stds = []
        for layout in layouts:
            row = final_summary[f"{layout}|{method}"]
            means.append(row["force_err_N_mean"] * 1e3)
            stds.append(row["force_err_N_std"] * 1e3)
        ax_force_bar.bar(
            x + (idx - 0.5) * width,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=METHOD_COLORS[method],
            alpha=0.85,
            label=METHOD_LABELS[method],
        )

    moment_methods = ["direct", "recursive_map", "moment_only"]
    for idx, method in enumerate(moment_methods):
        means = []
        stds = []
        for layout in layouts:
            row = final_summary[f"{layout}|{method}"]
            means.append(row["moment_err_Nm_mean"] * 1e3)
            stds.append(row["moment_err_Nm_std"] * 1e3)
        ax_moment_bar.bar(
            x + (idx - 1.0) * width,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=METHOD_COLORS[method],
            alpha=0.85,
            label=METHOD_LABELS[method],
        )

    for ax, ylabel, title in [
        (ax_force_bar, "Final force abs. error [mN]", "(a) Final force error"),
        (ax_moment_bar, "Final moment abs. error [mN·m]", "(c) Final moment error"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(layouts)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    # Stepwise mean error trajectories
    step_axis = np.arange(1, steps + 1)
    for layout in layouts:
        for method in force_methods:
            row = stepwise_summary[f"{layout}|{method}"]
            ax_force_step.plot(
                step_axis,
                row["force_err_traj_mean"] * 1e3,
                color=METHOD_COLORS[method],
                linestyle=LAYOUT_LINESTYLES[layout],
                linewidth=2.2,
                label=f"{METHOD_LABELS[method]} / {layout}",
            )
    for layout in layouts:
        for method in moment_methods:
            row = stepwise_summary[f"{layout}|{method}"]
            ax_moment_step.plot(
                step_axis,
                row["moment_err_traj_mean"] * 1e3,
                color=METHOD_COLORS[method],
                linestyle=LAYOUT_LINESTYLES[layout],
                linewidth=2.2,
                label=f"{METHOD_LABELS[method]} / {layout}",
            )

    for ax, ylabel, title in [
        (ax_force_step, "Mean force abs. error [mN]", "(b) Force error vs EKF step"),
        (ax_moment_step, "Mean moment abs. error [mN·m]", "(d) Moment error vs EKF step"),
    ]:
        ax.set_xlabel("EKF step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    handles, labels = ax_moment_step.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, framealpha=0.95)
    fig.suptitle("Tip-wrench estimator comparison: direct vs recursive MAP vs constrained", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    for ext in ("png", "pdf"):
        out_path = save_stem.parent / f"{save_stem.name}.{ext}"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"  FIG  -> {out_path}")
    plt.close(fig)


def evaluate_layout_methods(
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
    sigma_model_force: float,
    sigma_model_moment: float,
    sigma_rw_force: float,
    sigma_rw_moment: float,
    direct_rcond: float,
    moment_only_rcond: float,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Evaluate all methods for one layout.

    Returns
    -------
    final_rows
        one row per (case, noise realization, method), using final-step metrics
    stepwise_rows
        one row per (case, noise realization, method) containing trajectory arrays
    """
    order_cfg = BASELINE_ORDER
    num_cases, num_noise_real, n_imu, _, _ = R_meas.shape
    final_rows: List[Dict] = []
    stepwise_rows: List[Dict] = []
    Q_rw = np.diag([sigma_rw_moment**2] * 3 + [sigma_rw_force**2] * 3)
    t0 = time.perf_counter()

    for case_idx in range(num_cases):
        tau = tau_gt[case_idx]
        F_gt_world = np.hstack([l_ext_gt[case_idx], f_ext_gt[case_idx]])
        case_id = int(case_ids_gt[case_idx])

        for noise_idx in range(num_noise_real):
            R_frame = [R_meas[case_idx, noise_idx, sensor_idx] for sensor_idx in range(n_imu)]
            ekf_history = run_ekf_history_on_frame(
                R_frame=R_frame,
                imu_pos_norm=imu_pos_norm,
                e3=e3,
                gamma=gamma,
                meas_std_deg=meas_std_deg,
                alpha=alpha,
                P0=P0,
                steps=steps,
            )

            traj_metrics = {
                "direct": defaultdict(list),
                "recursive_map": defaultdict(list),
                "moment_only": defaultdict(list),
            }
            recursive_map_prev_world = None

            for step_idx, (m_est, P_est) in enumerate(ekf_history, start=1):
                terms = build_virtual_work_terms(
                    m=m_est,
                    tau=tau,
                    order_cfg=order_cfg,
                    gamma=gamma,
                    length_m=L_DEFAULT,
                    EIx=EIX_DEFAULT,
                    EIy=EIY_DEFAULT,
                    GJ=GJ_DEFAULT,
                )
                R_tip = terms.T_tip[:3, :3]

                # direct
                F_direct_body = direct_wrench_body_from_terms(terms, rcond=direct_rcond)
                F_direct_world = body_wrench_to_world(F_direct_body, R_tip)
                direct_metrics = wrench_metrics_from_world6(
                    F_direct_world, F_gt_world, estimate_force=True, estimate_moment=True
                )

                # recursive MAP in world frame
                Sigma_Fb = propagate_wrench_covariance_body(
                    terms,
                    P_est,
                    sigma_model_force=sigma_model_force,
                    sigma_model_moment=sigma_model_moment,
                    rcond=direct_rcond,
                    length_m=L_DEFAULT,
                    EIx=EIX_DEFAULT,
                    EIy=EIY_DEFAULT,
                    GJ=GJ_DEFAULT,
                )
                W = np.block(
                    [
                        [R_tip, np.zeros((3, 3))],
                        [np.zeros((3, 3)), R_tip],
                    ]
                )
                Sigma_Fw = W @ Sigma_Fb @ W.T
                if recursive_map_prev_world is None:
                    F_map_world = F_direct_world.copy()
                else:
                    F_map_world = recursive_map_update(
                        mean_meas=F_direct_world,
                        cov_meas=Sigma_Fw,
                        prior_mean=recursive_map_prev_world,
                        prior_cov=Q_rw,
                    )
                recursive_map_prev_world = F_map_world.copy()
                recursive_map_metrics = wrench_metrics_from_world6(
                    F_map_world, F_gt_world, estimate_force=True, estimate_moment=True
                )

                # observability-informed constrained estimator
                F_moment_only_body = moment_only_wrench_body_from_terms(
                    terms, rcond=moment_only_rcond
                )
                F_moment_only_world = body_wrench_to_world(F_moment_only_body, R_tip)
                moment_only_metrics = wrench_metrics_from_world6(
                    F_moment_only_world,
                    F_gt_world,
                    estimate_force=False,
                    estimate_moment=True,
                )

                payloads = {
                    "direct": (F_direct_world, direct_metrics, True, True),
                    "recursive_map": (F_map_world, recursive_map_metrics, True, True),
                    "moment_only": (F_moment_only_world, moment_only_metrics, False, True),
                }
                for method, (_, metrics, force_ok, moment_ok) in payloads.items():
                    traj_metrics[method]["force_err_traj"].append(metrics["force_err_N"])
                    traj_metrics[method]["moment_err_traj"].append(metrics["moment_err_Nm"])
                    traj_metrics[method]["force_dir_traj"].append(metrics["force_dir_err_deg"])
                    traj_metrics[method]["moment_dir_traj"].append(metrics["moment_dir_err_deg"])

                    if step_idx == steps:
                        row = {
                            "case_id": case_id,
                            "layout_name": layout_name,
                            "num_imus": n_imu,
                            "noise_real": noise_idx,
                            "method": method,
                            "steps": steps,
                            "force_metrics_applicable": force_ok,
                            "moment_metrics_applicable": moment_ok,
                        }
                        row.update(metrics)
                        final_rows.append(row)

            for method in METHOD_ORDER:
                stepwise_rows.append(
                    {
                        "case_id": case_id,
                        "layout_name": layout_name,
                        "num_imus": n_imu,
                        "noise_real": noise_idx,
                        "method": method,
                        "force_err_traj": np.array(traj_metrics[method]["force_err_traj"], dtype=float),
                        "moment_err_traj": np.array(traj_metrics[method]["moment_err_traj"], dtype=float),
                        "force_dir_traj": np.array(traj_metrics[method]["force_dir_traj"], dtype=float),
                        "moment_dir_traj": np.array(traj_metrics[method]["moment_dir_traj"], dtype=float),
                    }
                )

        if (case_idx + 1) % 10 == 0 or (case_idx + 1) == num_cases:
            elapsed = time.perf_counter() - t0
            print(f"    [{layout_name}] case {case_idx + 1:3d}/{num_cases}  elapsed {elapsed:.1f} s")

    return final_rows, stepwise_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare direct, corrected recursive MAP, and constrained wrench estimators.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt", default="gt_data/kirchhoff_gt_dataset.npz")
    parser.add_argument("--imu2", default="gt_data/kirchhoff_imu_2imu.npz")
    parser.add_argument("--imu3", default="gt_data/kirchhoff_imu_3imu.npz")
    parser.add_argument("--save-dir", default="gt_data/results_wrench_diagnostics")
    parser.add_argument("--gamma", type=int, default=GAMMA_DEFAULT)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--meas-std-deg", type=float, default=MEAS_STD_DEG_DEF)
    parser.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    parser.add_argument("--sigma-model-force", type=float, default=0.10)
    parser.add_argument("--sigma-model-moment", type=float, default=0.010)
    parser.add_argument(
        "--sigma-rw-force",
        type=float,
        default=0.05,
        help="Recursive MAP prior std dev for inter-step wrench change [N]",
    )
    parser.add_argument(
        "--sigma-rw-moment",
        type=float,
        default=0.005,
        help="Recursive MAP prior std dev for inter-step wrench change [N·m]",
    )
    parser.add_argument("--direct-rcond", type=float, default=1e-8)
    parser.add_argument("--moment-only-rcond", type=float, default=1e-8)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    gt_path = resolve_cli_path(args.gt, script_dir, must_exist=True)
    imu2_path = resolve_cli_path(args.imu2, script_dir, must_exist=False)
    imu3_path = resolve_cli_path(args.imu3, script_dir, must_exist=False)
    save_dir = resolve_cli_path(args.save_dir, script_dir, must_exist=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ground-truth dataset ...")
    positions_gt, orientations_gt, case_ids_gt, gt_meta = load_ground_truth_dataset(gt_path)
    gt_npz = np.load(gt_path, allow_pickle=True)
    tau_gt = gt_npz["tau"]
    f_ext_gt = gt_npz["f_ext"]
    l_ext_gt = gt_npz["l_ext"]
    print(f"  {len(case_ids_gt)} cases")

    print("\nLoading static IMU measurement files ...")
    imu_datasets: Dict[str, Dict] = {}
    for label, path in [("2-IMU", imu2_path), ("3-IMU", imu3_path)]:
        if not path.exists():
            print(f"  WARNING: {path} not found - skipping {label}")
            continue
        _, R_meas, _, imu_idx, imu_meta = load_imu_measurements(path)
        imu_actual_s = np.array(
            imu_meta.get(
                "imu_actual_s",
                (imu_idx / (positions_gt.shape[1] - 1)).tolist(),
            ),
            dtype=float,
        )
        imu_datasets[label] = {
            "R_meas": R_meas,
            "imu_pos": imu_actual_s,
        }
        print(
            f"  {label}: R_meas {R_meas.shape}, "
            + "{"
            + ", ".join(f"{value:.4f}" for value in imu_actual_s)
            + "}"
        )

    if not imu_datasets:
        raise RuntimeError("No IMU measurement files found for the comparison study.")

    P0 = np.diag(P0_DIAG)
    all_final_rows: List[Dict] = []
    all_stepwise_rows: List[Dict] = []

    for layout_name, ds in imu_datasets.items():
        print(f"\nEvaluating layout {layout_name} ...")
        final_rows, stepwise_rows = evaluate_layout_methods(
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
            sigma_model_force=args.sigma_model_force,
            sigma_model_moment=args.sigma_model_moment,
            sigma_rw_force=args.sigma_rw_force,
            sigma_rw_moment=args.sigma_rw_moment,
            direct_rcond=args.direct_rcond,
            moment_only_rcond=args.moment_only_rcond,
        )
        all_final_rows.extend(final_rows)
        all_stepwise_rows.extend(stepwise_rows)

    final_summary = aggregate_final_results(all_final_rows)
    stepwise_summary = aggregate_stepwise(all_stepwise_rows, steps=args.steps)

    print("\nEstimator comparison summary")
    sep = "=" * 92
    print(sep)
    print(
        f"{'Layout':<10}  {'Method':<14}  {'Force [mN]':>14}  "
        f"{'Moment [mN·m]':>16}  {'F-dir [deg]':>12}  {'M-dir [deg]':>12}"
    )
    print("-" * 92)
    for layout_name in imu_datasets.keys():
        for method in METHOD_ORDER:
            row = final_summary[f"{layout_name}|{method}"]
            force_str = (
                f"{row['force_err_N_mean'] * 1e3:>8.2f} +/- {row['force_err_N_std'] * 1e3:<8.2f}"
                if row["force_metrics_applicable"]
                else "   n/a        "
            )
            force_dir_str = (
                f"{row['force_dir_err_deg_mean']:>8.2f}"
                if row["force_metrics_applicable"]
                else "   n/a  "
            )
            print(
                f"{layout_name:<10}  {method:<14}  {force_str:>14}  "
                f"{row['moment_err_Nm_mean'] * 1e3:>8.3f} +/- {row['moment_err_Nm_std'] * 1e3:<8.3f}  "
                f"{force_dir_str:>12}  {row['moment_dir_err_deg_mean']:>12.2f}"
            )
    print(sep)

    save_csv(all_final_rows, save_dir / "kirchhoff_wrench_estimator_compare_results.csv")
    save_summary_csv(final_summary, save_dir / "kirchhoff_wrench_estimator_compare_summary.csv")
    save_json(
        {
            "config": {
                "workflow": "cross-model wrench comparison: direct vs recursive MAP vs moment-only",
                "gt": str(gt_path),
                "imu2": str(imu2_path),
                "imu3": str(imu3_path),
                "gamma": args.gamma,
                "alpha": args.alpha,
                "meas_std_deg": args.meas_std_deg,
                "steps": args.steps,
                "shape_model_order": list(BASELINE_ORDER),
                "sigma_model_force_N": args.sigma_model_force,
                "sigma_model_moment_Nm": args.sigma_model_moment,
                "sigma_rw_force_N": args.sigma_rw_force,
                "sigma_rw_moment_Nm": args.sigma_rw_moment,
                "recursive_map_note": (
                    "Recursive MAP uses F^-_k = F_hat_{k-1} in world-tip coordinates "
                    "with diagonal prior covariance Q_F."
                ),
                "constrained_method_note": (
                    "moment_only solves only body-tip moments and treats force metrics "
                    "as not applicable under the current 5D observability structure."
                ),
            },
            "final_summary": final_summary,
            "stepwise_summary": {
                key: {
                    sub_key: (
                        value.tolist() if isinstance(value, np.ndarray) else value
                    )
                    for sub_key, value in row.items()
                }
                for key, row in stepwise_summary.items()
            },
            "final_results": all_final_rows,
        },
        save_dir / "kirchhoff_wrench_estimator_compare_results.json",
    )
    save_stepwise_npz(
        stepwise_summary,
        save_dir / "kirchhoff_wrench_estimator_compare_stepwise.npz",
    )
    build_comparison_figure(
        final_summary=final_summary,
        stepwise_summary=stepwise_summary,
        steps=args.steps,
        save_stem=save_dir / "kirchhoff_wrench_estimator_compare",
    )


if __name__ == "__main__":
    main()
