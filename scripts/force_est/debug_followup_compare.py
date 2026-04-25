"""
Follow-up study: oracle (11-param) vs EKF wrench estimation.
Tasks 2 & 3 from the modal-basis follow-up request.

Runs on 3 representative cases from kirchhoff_gt_dataset.npz using:
  - 2-IMU layout: [0.5, 1.0]
  - 3-IMU layout: [0.25, 0.5, 1.0]

Compares oracle (GT-curvature) vs EKF for orders (1,1,0) and (3,3,2).
Does NOT modify any existing files.

Run from repo root:
  python scripts/force_est/debug_followup_compare.py
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRT  = ROOT / "src" / "shape_force_est_imu" / "crt"
sys.path.insert(0, str(CRT))
os.chdir(str(CRT))   # relative imports inside compare script work here

import numpy as np

from compare_oracle_vs_ekf_wrench_estimation import (
    estimate_modal_oracle,
    run_ekf_for_order,
    wrench_from_modal,
    wrench_metrics,
    fwd_transform_general,
    GAMMA_DEFAULT,
    MEAS_STD_DEG_DEF,
    ALPHA_DEFAULT,
    STEPS_DEFAULT,
)
from evaluate_kirchhoff_shape_estimation import (
    load_ground_truth_dataset,
    load_imu_measurements,
    compute_geometry_metrics,
    E3,
    so3_log,
)

OUT_DIR = ROOT / "scripts" / "force_est" / "debug_output" / "followup"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GT_PATH   = CRT / "gt_data" / "kirchhoff_gt_dataset.npz"
IMU2_PATH = CRT / "gt_data" / "kirchhoff_imu_2imu.npz"
IMU3_PATH = CRT / "gt_data" / "kirchhoff_imu_3imu.npz"

N_CASES  = 3     # only first 3 cases
N_NOISE  = 5     # all noise realizations available

ORDERS = [
    (1, 1, 0, "(1,1,0)"),
    (2, 2, 0, "(2,2,0)"),
    (2, 2, 1, "(2,2,1)"),
    (3, 3, 2, "(3,3,2)"),
]

def load_imu(path):
    R_true, R_meas, imu_pos, imu_idx, meta = load_imu_measurements(path)
    actual_s = np.array(meta.get("imu_actual_s",
                                 (imu_idx / 99.0).tolist()), dtype=float)
    return R_meas, actual_s   # (N, n_noise, n_imu, 3,3), (n_imu,)

def shape_metrics_from_modal(m, pos_gt_case, ori_gt_case, L, ox, oy, oz,
                              gamma=GAMMA_DEFAULT, n_pts=20):
    """Reconstruct backbone from modal state (arbitrary order) and compare to GT."""
    M_gt = len(pos_gt_case)
    # Use a coarser grid for speed; tip (s=1) is always included
    idx = np.round(np.linspace(0, M_gt - 1, n_pts)).astype(int)
    s_grid = np.linspace(0.0, 1.0, M_gt)[idx]
    p_est = np.zeros((n_pts, 3))
    R_est = np.zeros((n_pts, 3, 3))
    for i, s in enumerate(s_grid):
        T = fwd_transform_general(m, float(s), gamma, L, ox, oy, oz)
        p_est[i] = T[:3, 3]
        R_est[i] = T[:3, :3]
    R_gt = ori_gt_case.reshape(M_gt, 3, 3)[idx]
    g = compute_geometry_metrics(p_est, R_est, pos_gt_case[idx], R_gt)
    return {
        "tip_err_mm":  g["tip_position_error"] * 1e3,
        "rms_err_mm":  g["mean_centerline_error"] * 1e3,
    }

def run_case(ci, ox, oy, oz,
             pos_gt, ori_gt, tau, f_gt, l_gt,
             imu_datasets, L=0.1):
    row = {}
    # Oracle
    try:
        m_o = estimate_modal_oracle(pos_gt[ci], ori_gt[ci], ox, oy, oz, L)
        f_o, l_o = wrench_from_modal(m_o, tau[ci], ox, oy, oz)
        w = wrench_metrics(f_o, l_o, f_gt[ci], l_gt[ci])
        s = shape_metrics_from_modal(m_o, pos_gt[ci], ori_gt[ci], L, ox, oy, oz)
        row["oracle"] = {**w, **s}
    except Exception:
        row["oracle"] = None

    # EKF per layout
    for layout, (R_meas_all, imu_pos) in imu_datasets.items():
        errs = []
        for ni in range(N_NOISE):
            R_frame = [R_meas_all[ci, ni, si] for si in range(len(imu_pos))]
            try:
                m_e = run_ekf_for_order(
                    R_frame=R_frame, imu_pos_norm=imu_pos,
                    order_x=ox, order_y=oy, order_z=oz,
                    gamma=GAMMA_DEFAULT, meas_std_deg=MEAS_STD_DEG_DEF,
                    alpha=ALPHA_DEFAULT, steps=STEPS_DEFAULT,
                )
                f_e, l_e = wrench_from_modal(m_e, tau[ci], ox, oy, oz)
                w = wrench_metrics(f_e, l_e, f_gt[ci], l_gt[ci])
                s = shape_metrics_from_modal(m_e, pos_gt[ci], ori_gt[ci], L, ox, oy, oz)
                errs.append({**w, **s})
            except Exception:
                pass
        if errs:
            row[layout] = {
                k: float(np.mean([e[k] for e in errs]))
                for k in errs[0].keys()
            }
        else:
            row[layout] = None
    return row

def main():
    print("Loading datasets...")
    pos_gt, ori_gt, case_ids, gt_meta = load_ground_truth_dataset(GT_PATH)
    gt_np  = np.load(GT_PATH, allow_pickle=True)
    tau    = gt_np["tau"]
    f_gt   = gt_np["f_ext"]
    l_gt   = gt_np["l_ext"]
    L      = float(gt_meta.get("length_m", 0.1))

    R_meas2, pos2 = load_imu(IMU2_PATH)
    R_meas3, pos3 = load_imu(IMU3_PATH)
    imu_datasets = {
        "2-IMU [0.5,1.0]":       (R_meas2, pos2),
        "3-IMU [0.25,0.5,1.0]":  (R_meas3, pos3),
    }
    print(f"  GT: {pos_gt.shape[0]} cases, using first {N_CASES}")
    print(f"  2-IMU positions: {pos2}")
    print(f"  3-IMU positions: {pos3}")

    all_results = {}
    for ox, oy, oz, label in ORDERS:
        print(f"\n=== Order {label} ===")
        all_results[label] = {}
        for ci in range(N_CASES):
            print(f"  case {ci} ...", end=" ", flush=True)
            all_results[label][ci] = run_case(
                ci, ox, oy, oz,
                pos_gt, ori_gt, tau, f_gt, l_gt,
                imu_datasets, L,
            )
            print("done")

    # Save raw results
    def _ser(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    results_path = OUT_DIR / "followup_compare_results_v2.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_ser)
    print(f"\nSaved raw results -> {results_path}")

    # Print markdown summary
    print_summary(all_results, f_gt, l_gt, case_ids)

def print_summary(results, f_gt, l_gt, case_ids):
    metrics_to_show = [
        "tip_err_mm", "rms_err_mm",
        "force_dir_err_deg", "nrmse_force",
        "moment_dir_err_deg", "nrmse_moment",
    ]
    headers = ["Tip(mm)", "RMS(mm)", "F-dir(°)", "NRMSE-F", "M-dir(°)", "NRMSE-M"]
    methods = ["oracle", "2-IMU [0.5,1.0]", "3-IMU [0.25,0.5,1.0]"]

    summary_lines = []
    summary_lines.append("\n## Follow-up Study: Oracle vs EKF — Shape + Wrench\n")
    summary_lines.append(f"Dataset: kirchhoff_gt_dataset.npz (N={N_CASES} cases, 5 noise realizations)\n")

    for label in [o[3] for o in ORDERS]:
        summary_lines.append(f"\n### Order {label}\n")
        header = "| Method | " + " | ".join(headers) + " |"
        sep    = "|---|" + "|".join(["---"] * len(headers)) + "|"
        summary_lines.append(header)
        summary_lines.append(sep)

        order_data = results[label]
        for method in methods:
            vals = {m: [] for m in metrics_to_show}
            for ci in range(N_CASES):
                row = order_data[ci].get(method)
                if row is None:
                    continue
                for m in metrics_to_show:
                    if m in row:
                        vals[m].append(row[m])
            if not vals[metrics_to_show[0]]:
                na = " | ".join(["N/A"] * len(headers))
                summary_lines.append(f"| {method} | {na} |")
                continue
            cells = []
            for m, h in zip(metrics_to_show, headers):
                v = np.mean(vals[m]) if vals[m] else float("nan")
                if "deg" in m:
                    cells.append(f"{v:.1f}°")
                elif "mm" in m:
                    cells.append(f"{v:.2f}")
                else:
                    cells.append(f"{v:.3f}")
            summary_lines.append(f"| {method} | " + " | ".join(cells) + " |")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_path = OUT_DIR / "followup_compare_summary_v2.md"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\nSaved summary -> {summary_path}")

if __name__ == "__main__":
    main()
