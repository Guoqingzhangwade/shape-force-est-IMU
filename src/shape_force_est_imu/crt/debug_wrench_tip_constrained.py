"""
Debug: constrained tip-wrench solver on 3 representative force-only cases.

Loads 3 samples from gt_data_constrained/kirchhoff_gt_force_only.npz
(l_ext = 0 guaranteed, small / medium / large force magnitude).

Produces one composite figure (debug_constrained/debug_composite.pdf):
  Row 1 — 3D backbone: GT (black), oracle fit (green), EKF 2-IMU (blue)
           with tip force arrows for GT, oracle force_only, EKF force_only
  Row 2 — Force component bars [Fx, Fy, Fz]:
           GT | oracle force_only | EKF 2-IMU force_only | EKF 3-IMU force_only
           (error bars = std over 5 noise realisations for EKF)

IMU layouts:
  2-IMU: s = [0.5, 1.0]
  3-IMU: s = [0.25, 0.5, 1.0]

Usage
-----
  cd src/shape_force_est_imu/crt
  python debug_wrench_tip_constrained.py
  python debug_wrench_tip_constrained.py --lam 1e-3
  python debug_wrench_tip_constrained.py --cases 48 43 11
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.linalg import expm

from evaluate_kirchhoff_shape_estimation import (
    ALPHA_DEFAULT, E3, GAMMA_DEFAULT, MEAS_STD_DEG_DEF,
    P0_DIAG, STEPS_DEFAULT, run_ekf_on_frame, skew,
)
from compare_oracle_vs_ekf_wrench_estimation import estimate_modal_oracle
from wrench_tip_constrained_solver import (
    LAMBDA_DEFAULT, _ORDER_X, _ORDER_Y, _ORDER_Z, _GAMMA,
    make_S_direction, make_S_force_only,
    solve_constrained_wrench, world_wrench_from_body, wrench_metrics,
)
from virtual_work import body_jacobian_at_s, product_of_exponentials

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_DEBUG_CASES  = [48, 43, 11]        # small / medium / large force
_CASE_LABELS  = ["Small (0.11 N)", "Medium (0.41 N)", "Large (0.59 N)"]
_IMU_LAYOUTS  = {
    "2-IMU": np.array([0.5, 1.0]),
    "3-IMU": np.array([0.25, 0.5, 1.0]),
}
_MEAS_STD_DEG = 0.5
_N_NOISE      = 5

# Plot colours
_C_GT     = "#111111"
_C_ORA    = "#2ca02c"   # green
_C_EKF2   = "#1f77b4"   # blue
_C_EKF3   = "#9467bd"   # purple
_C_KNOWN  = "#d62728"   # red (known_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reconstruct_backbone(m, n_pts=100, gamma=_GAMMA,
                          order_x=_ORDER_X, order_y=_ORDER_Y,
                          order_z=_ORDER_Z, L=0.1):
    """(n_pts, 3) backbone positions. product_of_exponentials already scales by L."""
    s_vals = np.linspace(0.0, 1.0, n_pts)
    pts = np.zeros((n_pts, 3))
    for i, s in enumerate(s_vals):
        T = product_of_exponentials(m, s, gamma, L, order_x, order_y, order_z)
        pts[i] = T[:3, 3]
    return pts


def synthesize_R_meas(orientations_gt, imu_arc_norm, num_pts,
                       n_noise, meas_std_deg, rng):
    sigma = np.deg2rad(meas_std_deg)
    n_imu = len(imu_arc_norm)
    arc_indices = [int(round(s * (num_pts - 1))) for s in imu_arc_norm]
    R_meas = np.zeros((n_noise, n_imu, 3, 3))
    for ni in range(n_noise):
        for ii, idx in enumerate(arc_indices):
            R_true = orientations_gt[idx].reshape(3, 3)
            eta    = rng.standard_normal(3) * sigma
            R_meas[ni, ii] = expm(skew(eta)) @ R_true
    return R_meas, arc_indices


# ---------------------------------------------------------------------------
# Main evaluation — returns per-case result dicts
# ---------------------------------------------------------------------------

def evaluate_cases(args):
    rng        = np.random.default_rng(args.seed)
    script_dir = Path(__file__).resolve().parent
    gt_path    = script_dir / "gt_data_constrained" / "kirchhoff_gt_force_only.npz"
    save_dir   = script_dir / "debug_constrained"
    save_dir.mkdir(exist_ok=True)

    data            = np.load(gt_path, allow_pickle=True)
    positions_gt    = data["positions"]      # (N, 100, 3)
    orientations_gt = data["orientations"]   # (N, 100, 9)
    tau_all         = data["tau"]            # (N, 4)
    f_ext_all       = data["f_ext"]          # (N, 3)  world frame
    l_ext_all       = data["l_ext"]          # (N, 3)  all zero
    case_ids        = data["case_id"]
    num_pts         = positions_gt.shape[1]

    assert np.allclose(l_ext_all, 0), "Expected l_ext=0 for force-only dataset"

    S_force = make_S_force_only()   # 6×3

    all_rows: List[Dict] = []
    case_results = []   # one dict per case, holds arrays for plotting

    for ci, label in zip(args.cases, _CASE_LABELS):
        print(f"\n=== Case {ci}  [{label}] ===")
        pos_gt = positions_gt[ci]
        ori_gt = orientations_gt[ci]
        tau    = tau_all[ci]
        f_gt   = f_ext_all[ci]          # world-frame GT force
        l_gt   = l_ext_all[ci]          # zero

        # Known-direction S (oracle cheat: uses GT force direction in body frame)
        R_tip_gt  = ori_gt[-1].reshape(3, 3)
        d_world   = f_gt / (np.linalg.norm(f_gt) + 1e-12)
        d_body    = R_tip_gt.T @ d_world
        S_dir     = make_S_direction(d_body)   # 6×1

        # ---- Oracle fit ----
        m_oracle   = estimate_modal_oracle(pos_gt, ori_gt, _ORDER_X, _ORDER_Y, _ORDER_Z)
        pos_oracle = reconstruct_backbone(m_oracle)
        tip_err_mm = np.linalg.norm(pos_gt[-1] - pos_oracle[-1]) * 1e3

        # Oracle wrench estimates
        w_fo, _, _, T_fo = solve_constrained_wrench(m_oracle, tau, S_force, lam=args.lam)
        f_ora_fo, _ = world_wrench_from_body(w_fo, T_fo)
        w_kd, _, _, T_kd = solve_constrained_wrench(m_oracle, tau, S_dir, lam=args.lam)
        f_ora_kd, _ = world_wrench_from_body(w_kd, T_kd)

        met_ora_fo = wrench_metrics(f_ora_fo, np.zeros(3), f_gt, l_gt)
        met_ora_kd = wrench_metrics(f_ora_kd, np.zeros(3), f_gt, l_gt)
        print(f"  Oracle tip err: {tip_err_mm:.2f} mm")
        print(f"  Oracle force_only:  NRMSE={met_ora_fo['nrmse_force']*100:.1f}%  "
              f"dir={met_ora_fo['force_dir_err_deg']:.1f}°")
        print(f"  Oracle known_dir:   NRMSE={met_ora_kd['nrmse_force']*100:.1f}%  "
              f"dir={met_ora_kd['force_dir_err_deg']:.1f}°")

        # ---- EKF per layout ----
        ekf_backbones = {}     # layout → (n_pts, 3) mean backbone
        ekf_f_fo      = {}     # layout → (n_noise, 3) estimated forces
        ekf_f_kd      = {}     # layout → (n_noise, 3) estimated forces

        for layout_name, imu_arc in _IMU_LAYOUTS.items():
            R_meas, _ = synthesize_R_meas(ori_gt, imu_arc, num_pts,
                                           _N_NOISE, _MEAS_STD_DEG, rng)
            m_runs = []
            for ni in range(_N_NOISE):
                R_frame = [R_meas[ni, ii] for ii in range(len(imu_arc))]
                try:
                    m_est, _ = run_ekf_on_frame(
                        R_frame=R_frame, imu_pos_norm=imu_arc,
                        e3=E3.copy(), gamma=_GAMMA,
                        meas_std_deg=_MEAS_STD_DEG, alpha=ALPHA_DEFAULT,
                        P0=np.diag(P0_DIAG), steps=STEPS_DEFAULT,
                    )
                except Exception:
                    m_est = np.zeros_like(m_oracle)
                m_runs.append(m_est)

            ekf_backbones[layout_name] = np.mean(
                [reconstruct_backbone(m) for m in m_runs], axis=0)

            tip_ekf_mm = np.linalg.norm(
                pos_gt[-1] - ekf_backbones[layout_name][-1]) * 1e3
            print(f"  EKF [{layout_name}] tip err: {tip_ekf_mm:.2f} mm")

            f_fo_runs, f_kd_runs = [], []
            for m_est in m_runs:
                w_fo_e, _, _, T_fo_e = solve_constrained_wrench(
                    m_est, tau, S_force, lam=args.lam)
                f_fo_e, _ = world_wrench_from_body(w_fo_e, T_fo_e)
                f_fo_runs.append(f_fo_e)

                w_kd_e, _, _, T_kd_e = solve_constrained_wrench(
                    m_est, tau, S_dir, lam=args.lam)
                f_kd_e, _ = world_wrench_from_body(w_kd_e, T_kd_e)
                f_kd_runs.append(f_kd_e)

            ekf_f_fo[layout_name] = np.array(f_fo_runs)   # (n_noise, 3)
            ekf_f_kd[layout_name] = np.array(f_kd_runs)

            for sc_name, f_runs in [("force_only", f_fo_runs), ("known_dir", f_kd_runs)]:
                for ni, f_e in enumerate(f_runs):
                    met = wrench_metrics(f_e, np.zeros(3), f_gt, l_gt)
                    row = {"case_idx": ci, "solver_case": sc_name,
                           "method": f"ekf_{layout_name}",
                           "layout": layout_name, "noise_real": ni}
                    row.update(met)
                    all_rows.append(row)

            mean_fo = np.mean([m["force_err_N"] for m in
                               [wrench_metrics(f, np.zeros(3), f_gt, l_gt)
                                for f in f_fo_runs]]) * 1e3
            mean_kd = np.mean([m["force_err_N"] for m in
                               [wrench_metrics(f, np.zeros(3), f_gt, l_gt)
                                for f in f_kd_runs]]) * 1e3
            print(f"    force_only: F_err={mean_fo:.1f} mN   "
                  f"known_dir: F_err={mean_kd:.1f} mN")

        # Store oracle rows
        for sc_name, f_est, met in [
            ("force_only", f_ora_fo, met_ora_fo),
            ("known_dir",  f_ora_kd, met_ora_kd),
        ]:
            row = {"case_idx": ci, "solver_case": sc_name, "method": "oracle",
                   "layout": "GT", "noise_real": -1}
            row.update(met)
            all_rows.append(row)

        case_results.append({
            "ci": ci, "label": label,
            "pos_gt": pos_gt,
            "pos_oracle": pos_oracle,
            "pos_ekf2": ekf_backbones["2-IMU"],
            "pos_ekf3": ekf_backbones["3-IMU"],
            "f_gt":         f_gt,
            "f_ora_fo":     f_ora_fo,
            "f_ora_kd":     f_ora_kd,
            "f_ekf2_fo":    ekf_f_fo["2-IMU"],   # (n_noise, 3)
            "f_ekf3_fo":    ekf_f_fo["3-IMU"],
            "f_ekf2_kd":    ekf_f_kd["2-IMU"],
            "f_ekf3_kd":    ekf_f_kd["3-IMU"],
            "imu_pts_2": np.array([
                pos_gt[int(round(s * (num_pts - 1)))]
                for s in _IMU_LAYOUTS["2-IMU"]]),
            "tip_err_oracle_mm": tip_err_mm,
        })

    # ---- CSV ----
    if all_rows:
        csv_path = save_dir / "debug_constrained_results.csv"
        keys = list(all_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(all_rows)
        print(f"\nCSV -> {csv_path}")

    # ---- Console summary ----
    _print_summary(all_rows)

    return case_results, save_dir, all_rows


# ---------------------------------------------------------------------------
# Composite figure
# ---------------------------------------------------------------------------

def plot_composite(case_results, save_dir: Path) -> None:
    n = len(case_results)
    fig = plt.figure(figsize=(5.5 * n, 10))
    fig.suptitle(
        "Constrained tip-wrench debug  |  force-only GT dataset  |  λ=1e-4",
        fontsize=12, y=0.99,
    )

    axes_3d  = []
    axes_bar = []
    for col in range(n):
        ax3 = fig.add_subplot(2, n, col + 1, projection="3d")
        axb = fig.add_subplot(2, n, n + col + 1)
        axes_3d.append(ax3)
        axes_bar.append(axb)

    for col, cd in enumerate(case_results):
        ax3 = axes_3d[col]
        axb = axes_bar[col]

        # ---- Row 1: 3D backbone + force arrows ----
        pg = cd["pos_gt"]
        po = cd["pos_oracle"]
        pe = cd["pos_ekf2"]

        ax3.plot(*pg.T, color=_C_GT,   lw=1.8, label="GT",        zorder=3)
        ax3.plot(*po.T, color=_C_ORA,  lw=1.4, ls="--", label="Oracle", zorder=2)
        ax3.plot(*pe.T, color=_C_EKF2, lw=1.2, ls=":",  label="EKF 2-IMU", zorder=2)

        # IMU sensor positions
        for ip in cd["imu_pts_2"]:
            ax3.scatter(*ip, color=_C_EKF2, s=30, marker="D", zorder=5)

        tip = pg[-1]
        f_gt  = cd["f_gt"]
        f_ofo = cd["f_ora_fo"]
        f_eof = cd["f_ekf2_fo"].mean(axis=0)

        # Scale arrows to 30% of rod length
        scale = 0.1 * 0.30 / (np.linalg.norm(f_gt) + 1e-9)
        ax3.quiver(*tip, *f_gt  * scale, color=_C_GT,   lw=2,   arrow_length_ratio=0.3, label="GT force")
        ax3.quiver(*tip, *f_ofo * scale, color=_C_ORA,  lw=1.6, arrow_length_ratio=0.3, label="Oracle est.")
        ax3.quiver(*tip, *f_eof * scale, color=_C_EKF2, lw=1.4, arrow_length_ratio=0.3, label="EKF est.")

        ax3.set_title(cd["label"], fontsize=10, pad=4)
        ax3.set_xlabel("x [m]", fontsize=7, labelpad=1)
        ax3.set_ylabel("y [m]", fontsize=7, labelpad=1)
        ax3.set_zlabel("z [m]", fontsize=7, labelpad=1)
        ax3.tick_params(labelsize=6)
        ax3.set_box_aspect([1, 1, 1])
        if col == 0:
            ax3.legend(fontsize=6, loc="upper left", framealpha=0.7)

        # Annotate tip errors
        tip_err = cd["tip_err_oracle_mm"]
        ax3.text2D(0.02, 0.02,
                   f"Oracle tip Δ={tip_err:.1f} mm",
                   transform=ax3.transAxes, fontsize=6, color=_C_ORA)

        # ---- Row 2: force components bar chart (force_only solver) ----
        comp_labels = ["Fx", "Fy", "Fz"]
        x    = np.arange(3)
        w    = 0.18
        f_g  = cd["f_gt"]
        f_of = cd["f_ora_fo"]
        f_e2 = cd["f_ekf2_fo"]   # (5, 3)
        f_e3 = cd["f_ekf3_fo"]   # (5, 3)

        # Convert to mN
        scale_mn = 1e3
        bars = [
            ("GT",          f_g * scale_mn,           None,                         _C_GT),
            ("Oracle",      f_of * scale_mn,           None,                        _C_ORA),
            ("EKF 2-IMU",   f_e2.mean(0) * scale_mn,  f_e2.std(0) * scale_mn,      _C_EKF2),
            ("EKF 3-IMU",   f_e3.mean(0) * scale_mn,  f_e3.std(0) * scale_mn,      _C_EKF3),
        ]
        for bi, (lbl, vals, errs, color) in enumerate(bars):
            offset = (bi - 1.5) * w
            axb.bar(x + offset, vals, width=w, color=color,
                    alpha=0.85, label=lbl, zorder=3)
            if errs is not None:
                axb.errorbar(x + offset, vals, yerr=errs,
                             fmt="none", color="k", capsize=2, lw=0.8, zorder=4)

        axb.axhline(0, color="k", lw=0.6, ls="-")
        axb.set_xticks(x)
        axb.set_xticklabels(comp_labels, fontsize=9)
        axb.set_ylabel("Force [mN]", fontsize=8)
        axb.set_title(f"{cd['label']}  —  force_only solver", fontsize=9)
        axb.tick_params(labelsize=7)
        axb.grid(axis="y", lw=0.4, alpha=0.5)
        if col == 0:
            axb.legend(fontsize=7, framealpha=0.8)

        # NRMSE annotation
        met_gt_fo = wrench_metrics(cd["f_ora_fo"], np.zeros(3), cd["f_gt"], np.zeros(3))
        met_ek_fo = wrench_metrics(cd["f_ekf2_fo"].mean(0), np.zeros(3), cd["f_gt"], np.zeros(3))
        axb.text(0.02, 0.97,
                 f"Oracle NRMSE: {met_gt_fo['nrmse_force']*100:.1f}%\n"
                 f"EKF 2-IMU NRMSE: {met_ek_fo['nrmse_force']*100:.1f}%",
                 transform=axb.transAxes, fontsize=7, va="top",
                 color="k", bbox=dict(fc="white", ec="none", alpha=0.7))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "png"):
        p = save_dir / f"debug_composite.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Figure -> {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(all_rows):
    print("\n" + "=" * 84)
    print("  SUMMARY  (mean ± std over noise realisations, force-only solver)")
    print("=" * 84)
    buckets = defaultdict(list)
    for r in all_rows:
        buckets[(r["case_idx"], r["solver_case"], r["method"])].append(r)
    hdr = f"{'case':>5}  {'solver':>12}  {'method':>14}  " \
          f"{'F_err[mN]':>13}  {'NRMSE-F':>9}  {'dir[deg]':>9}"
    print(hdr)
    print("-" * 84)
    for (ci, sc, method), rows in sorted(buckets.items()):
        ferrs  = [r["force_err_N"] * 1e3 for r in rows]
        nrmses = [r["nrmse_force"] * 100  for r in rows
                  if not np.isnan(r["nrmse_force"])]
        dirs   = [r["force_dir_err_deg"] for r in rows]
        print(f"{ci:>5}  {sc:>12}  {method:>14}  "
              f"{np.mean(ferrs):>8.1f}±{np.std(ferrs):<4.1f}  "
              f"{np.nanmean(nrmses) if nrmses else float('nan'):>8.1f}%  "
              f"{np.mean(dirs):>8.1f}°")
    print("=" * 84)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Debug constrained tip-wrench solver — force-only dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cases", nargs="+", type=int, default=_DEBUG_CASES,
                        help="Row indices into kirchhoff_gt_force_only.npz")
    parser.add_argument("--lam",  type=float, default=LAMBDA_DEFAULT,
                        help="Regularisation strength lambda")
    parser.add_argument("--seed", type=int,   default=42)
    args = parser.parse_args()

    # Pad _CASE_LABELS if user supplies different cases
    global _CASE_LABELS
    if len(args.cases) != len(_CASE_LABELS):
        _CASE_LABELS = [f"Case {c}" for c in args.cases]

    case_results, save_dir, _ = evaluate_cases(args)
    print("\nGenerating composite figure...")
    plot_composite(case_results, save_dir)
    print(f"\nAll outputs in: {save_dir}")


if __name__ == "__main__":
    main()
