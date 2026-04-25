#!/usr/bin/env python3
"""
Cross-model robustness study - Step 3b: EKF evaluation with measurement sequences.

This script intentionally leaves the original cross-model evaluator untouched.

Methodological distinction
--------------------------
  - Original Step 3 (`evaluate_kirchhoff_shape_estimation.py`)
    loads one saved noisy IMU frame per realization and repeats that same frame
    at every EKF step.

  - New Step 3b (this script)
    loads a full saved sequence of noisy IMU frames and feeds
    `R_meas_seq[case, realization, step]` to the EKF step-by-step.

In both workflows the underlying Kirchhoff-rod ground-truth shape is static.
Only the measurement handling differs: repeated static observation versus
time-varying noisy IMU observations.

Saved outputs (in --save-dir)
-----------------------------
  kirchhoff_shape_est_results_seq.csv
  kirchhoff_shape_est_summary_by_layout_seq.csv
  kirchhoff_shape_est_results_seq.json
  kirchhoff_shape_est_shapes_seq.npz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from evaluate_kirchhoff_shape_estimation import (
    ALPHA_DEFAULT,
    E3,
    GAMMA_DEFAULT,
    MEAS_STD_DEG_DEF,
    P0_DIAG,
    Q_DIAG,
    STATE_DIM,
    aggregate_by_layout,
    compute_geometry_metrics,
    load_ground_truth_dataset,
    plot_cross_model_summary,
    plot_representative_overlays,
    reconstruct_shape,
    save_results_csv,
    save_results_json,
    save_shapes_npz,
    save_summary_csv,
    so3_analytic_H_and_r,
)


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


def load_imu_measurement_sequences(
    npz_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Returns:
      case_ids, R_true, R_meas_seq, imu_positions, imu_arc_indices, meta
    """
    data = np.load(npz_path, allow_pickle=True)
    case_ids = data["case_id"]
    R_true = data["R_true"]
    R_meas_seq = data["R_meas_seq"]
    imu_positions = data["imu_positions"]
    imu_arc_indices = data["imu_arc_indices"]
    meta: dict = {}
    if "meta" in data:
        try:
            meta = json.loads(str(data["meta"].item()))
        except Exception:
            pass
    return case_ids, R_true, R_meas_seq, imu_positions, imu_arc_indices, meta


def run_ekf_on_sequence(
    R_seq: np.ndarray,
    imu_pos_norm: np.ndarray,
    e3: np.ndarray,
    gamma: int,
    meas_std_deg: float,
    alpha: float,
    P0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the same SO(3)-analytic EKF as the original cross-model evaluator,
    but feed one saved noisy IMU frame per EKF step.

    Parameters
    ----------
    R_seq         : (n_steps, n_imu, 3, 3)
    imu_pos_norm  : (n_imu,)
    """
    sigma = np.deg2rad(meas_std_deg)
    R_sngl = alpha * sigma**2 * np.eye(3)
    Q = np.diag(Q_DIAG)
    n_imu = len(imu_pos_norm)
    R_big = np.kron(np.eye(n_imu), R_sngl)

    m_est = np.zeros(STATE_DIM)
    P_est = P0.copy()

    for step_idx in range(R_seq.shape[0]):
        R_frame = [R_seq[step_idx, sensor_idx] for sensor_idx in range(n_imu)]
        P_pred = P_est + Q
        H, r = so3_analytic_H_and_r(m_est, imu_pos_norm, R_frame, e3, gamma)
        innov = -r
        S = H @ P_pred @ H.T + R_big
        K = P_pred @ H.T @ np.linalg.inv(S)
        m_est = m_est + K @ innov
        IKH = np.eye(STATE_DIM) - K @ H
        P_est = IKH @ P_pred @ IKH.T + K @ R_big @ K.T
        P_est = 0.5 * (P_est + P_est.T)

    return m_est, P_est


def evaluate_layout_sequences(
    positions_gt: np.ndarray,
    orientations_gt: np.ndarray,
    case_ids_gt: np.ndarray,
    case_ids_meas: np.ndarray,
    R_meas_seq: np.ndarray,
    imu_pos_norm: np.ndarray,
    s_grid: np.ndarray,
    L_phys: float,
    e3: np.ndarray,
    gamma: int,
    meas_std_deg: float,
    alpha: float,
    P0: np.ndarray,
    layout_name: str,
) -> List[Dict]:
    """
    Run the EKF for all cases x noise realizations for one sensor layout.

    Returns per-run result dicts. Private keys prefixed with '_' are kept for
    the saved shapes NPZ and stripped before JSON/CSV export.
    """
    if not np.array_equal(case_ids_gt, case_ids_meas):
        raise ValueError(
            f"Case-id mismatch between GT dataset and {layout_name} measurement sequence file."
        )

    num_cases, num_noise_realizations, num_steps, n_imu, _, _ = R_meas_seq.shape
    num_pts = positions_gt.shape[1]
    R_gt_all = orientations_gt.reshape(num_cases, num_pts, 3, 3)

    results: List[Dict] = []
    t0 = time.perf_counter()

    for case_idx in range(num_cases):
        p_gt = positions_gt[case_idx]
        R_gt = R_gt_all[case_idx]

        for noise_idx in range(num_noise_realizations):
            m_est, _ = run_ekf_on_sequence(
                R_seq=R_meas_seq[case_idx, noise_idx],
                imu_pos_norm=imu_pos_norm,
                e3=e3,
                gamma=gamma,
                meas_std_deg=meas_std_deg,
                alpha=alpha,
                P0=P0,
            )

            p_est, R_est = reconstruct_shape(m_est, s_grid, e3, gamma, L_phys)
            metrics = compute_geometry_metrics(p_est, R_est, p_gt, R_gt)

            results.append(
                {
                    "case_id": int(case_ids_gt[case_idx]),
                    "layout_name": layout_name,
                    "num_imus": n_imu,
                    "noise_realization": noise_idx,
                    "num_steps": num_steps,
                    **metrics,
                    "_m_est": m_est,
                    "_p_est": p_est,
                    "_R_est": R_est,
                }
            )

        if (case_idx + 1) % 10 == 0 or (case_idx + 1) == num_cases:
            elapsed = time.perf_counter() - t0
            print(
                f"    [{layout_name}] case {case_idx + 1:3d}/{num_cases}  "
                f"elapsed {elapsed:.1f} s"
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3b: EKF robustness evaluation with time-varying IMU sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gt",
        default="gt_data/kirchhoff_gt_dataset.npz",
        help="Step-1 ground-truth NPZ",
    )
    parser.add_argument(
        "--imu2",
        default="gt_data_seq/kirchhoff_imu_seq_2imu.npz",
        help="Step-2b 2-IMU measurement-sequence NPZ",
    )
    parser.add_argument(
        "--imu3",
        default="gt_data_seq/kirchhoff_imu_seq_3imu.npz",
        help="Step-2b 3-IMU measurement-sequence NPZ",
    )
    parser.add_argument(
        "--save-dir",
        default="gt_data_seq/results_seq",
        help="directory for sequence-based evaluation outputs",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="optional expected sequence length; if set, must match the saved files",
    )
    parser.add_argument("--gamma", type=int, default=GAMMA_DEFAULT)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--meas-std-deg", type=float, default=MEAS_STD_DEG_DEF)
    parser.add_argument(
        "--plot-summary",
        action="store_true",
        help="show and save aggregate comparison plot",
    )
    parser.add_argument(
        "--plot-overlays",
        action="store_true",
        help="show and save representative shape overlays",
    )
    parser.add_argument("--num-overlay-cases", type=int, default=3)
    args = parser.parse_args()

    if args.meas_std_deg <= 0.0:
        raise ValueError("--meas-std-deg must be positive.")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive when provided.")

    script_dir = Path(__file__).resolve().parent
    gt_path = resolve_cli_path(args.gt, script_dir, must_exist=True)
    imu2_path = resolve_cli_path(args.imu2, script_dir, must_exist=False)
    imu3_path = resolve_cli_path(args.imu3, script_dir, must_exist=False)
    save_dir = resolve_cli_path(args.save_dir, script_dir, must_exist=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ground-truth dataset ...")
    positions_gt, orientations_gt, case_ids_gt, gt_meta = load_ground_truth_dataset(gt_path)
    num_cases, num_pts, _ = positions_gt.shape
    L_phys = float(gt_meta.get("length_m", 0.10))
    s_grid = np.linspace(0.0, 1.0, num_pts)
    print(f"  {num_cases} cases, {num_pts} arc-length points, L = {L_phys} m")

    print("\nLoading measurement-sequence files ...")
    imu_datasets: Dict[str, Dict] = {}
    expected_steps = args.steps

    for label, path in [("2-IMU", imu2_path), ("3-IMU", imu3_path)]:
        if not path.exists():
            print(f"  WARNING: {path} not found - skipping {label}")
            continue

        case_ids_meas, R_true, R_meas_seq, imu_pos, imu_idx, imu_meta = (
            load_imu_measurement_sequences(path)
        )
        sequence_steps = int(R_meas_seq.shape[2])
        if expected_steps is not None and sequence_steps != expected_steps:
            raise ValueError(
                f"{label} sequence length mismatch: file has {sequence_steps} steps, "
                f"but --steps={expected_steps}."
            )
        if expected_steps is None:
            expected_steps = sequence_steps
        elif sequence_steps != expected_steps:
            raise ValueError(
                f"Loaded sequence files do not agree on step count: "
                f"expected {expected_steps}, got {sequence_steps} for {label}."
            )

        imu_actual_s = np.array(
            imu_meta.get("imu_actual_s", (imu_idx / (num_pts - 1)).tolist()),
            dtype=float,
        )
        imu_datasets[label] = {
            "case_ids": case_ids_meas,
            "R_true": R_true,
            "R_meas_seq": R_meas_seq,
            "imu_pos": imu_actual_s,
            "imu_idx": imu_idx,
            "sequence_steps": sequence_steps,
            "meta": imu_meta,
            "source_path": str(path),
        }
        pos_str = "{" + ", ".join(f"{value:.4f}" for value in imu_actual_s) + "}"
        print(
            f"  {label}: R_meas_seq {R_meas_seq.shape}, "
            f"imu_pos = {pos_str}, steps = {sequence_steps}"
        )

    if not imu_datasets:
        raise RuntimeError(
            "No measurement-sequence files found. Run Step 2b first."
        )

    P0 = np.diag(P0_DIAG)
    all_results: List[Dict] = []

    for layout_name, ds in imu_datasets.items():
        print(f"\nEvaluating {layout_name} with time-varying IMU sequences ...")
        layout_results = evaluate_layout_sequences(
            positions_gt=positions_gt,
            orientations_gt=orientations_gt,
            case_ids_gt=case_ids_gt,
            case_ids_meas=ds["case_ids"],
            R_meas_seq=ds["R_meas_seq"],
            imu_pos_norm=ds["imu_pos"],
            s_grid=s_grid,
            L_phys=L_phys,
            e3=E3.copy(),
            gamma=args.gamma,
            meas_std_deg=args.meas_std_deg,
            alpha=args.alpha,
            P0=P0,
            layout_name=layout_name,
        )
        all_results.extend(layout_results)
        print(f"  {len(layout_results)} runs complete.")

    summary = aggregate_by_layout(all_results)

    sep = "=" * 78
    print(f"\n{sep}")
    print(
        f"  Cross-model robustness with time-varying IMU sequences  "
        f"(alpha={args.alpha}, steps={expected_steps}, gamma={args.gamma})"
    )
    print(sep)
    print(
        f"{'Layout':<10}  {'n_IMU':>5}  "
        f"{'mean_pos [mm]':>14}  {'tip_pos [mm]':>13}  "
        f"{'tip_ori [deg]':>14}"
    )
    print("-" * 78)
    for name, stats in summary.items():
        print(
            f"{name:<10}  {stats['num_imus']:>5}  "
            f"{stats['mean_centerline_error_mean'] * 1e3:>10.3f}"
            f" +/-{stats['mean_centerline_error_std'] * 1e3:<6.3f}"
            f"{stats['tip_position_error_mean'] * 1e3:>10.3f}"
            f" +/-{stats['tip_position_error_std'] * 1e3:<6.3f}"
            f"{stats['tip_orientation_error_deg_mean']:>10.3f}"
            f" +/-{stats['tip_orientation_error_deg_std']:.3f}"
        )
    print(sep)

    results_csv_path = save_dir / "kirchhoff_shape_est_results_seq.csv"
    summary_csv_path = save_dir / "kirchhoff_shape_est_summary_by_layout_seq.csv"
    results_json_path = save_dir / "kirchhoff_shape_est_results_seq.json"
    shapes_npz_path = save_dir / "kirchhoff_shape_est_shapes_seq.npz"

    save_results_csv(all_results, results_csv_path)
    save_summary_csv(summary, summary_csv_path)
    save_results_json(
        all_results,
        summary,
        cfg={
            "workflow": "sequence-based cross-model robustness evaluation",
            "method_note": (
                "Original cross-model Step 3 reused one static noisy IMU frame at "
                "every EKF step. This Step 3b evaluator consumes a full saved noisy "
                "IMU sequence, one frame per EKF step."
            ),
            "gt": str(gt_path),
            "imu2": str(imu2_path),
            "imu3": str(imu3_path),
            "steps": expected_steps,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "meas_std_deg": args.meas_std_deg,
            "P0_diag": P0_DIAG.tolist(),
            "Q_diag": Q_DIAG.tolist(),
            "L_phys": L_phys,
            "num_pts": num_pts,
        },
        path=results_json_path,
    )
    save_shapes_npz(all_results, list(imu_datasets.keys()), shapes_npz_path)

    plot_stem = save_dir / "kirchhoff_shape_est_seq"
    if args.plot_summary:
        plot_cross_model_summary(summary, save_stem=plot_stem)
    if args.plot_overlays:
        plot_representative_overlays(
            positions_gt=positions_gt,
            case_ids=case_ids_gt,
            all_results=all_results,
            layout_names=list(imu_datasets.keys()),
            n_cases=args.num_overlay_cases,
            save_stem=plot_stem,
        )

    print("\nWorkflow distinction")
    print("  Original static-frame workflow : one noisy IMU frame repeated across EKF steps.")
    print("  New sequence workflow          : one noisy IMU frame per saved EKF step.")


if __name__ == "__main__":
    main()
