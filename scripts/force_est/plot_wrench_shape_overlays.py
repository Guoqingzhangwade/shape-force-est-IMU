"""
Shape overlay plot with tip-force vectors.

For each of 4 modal orders × 3 cases:
  - GT backbone (black solid)
  - Oracle backbone (green dashed)
  - EKF 2-IMU backbone (blue dashed)
  - EKF 3-IMU backbone (red dashed)
  - Tip force arrows (GT black, oracle green, EKF-2IMU blue, EKF-3IMU red)

Layout: 4 rows (orders) × 3 cols (cases).

Run from repo root:
  python scripts/force_est/plot_wrench_shape_overlays.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRT  = ROOT / "src" / "shape_force_est_imu" / "crt"
sys.path.insert(0, str(CRT))
os.chdir(str(CRT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from evaluate_kirchhoff_shape_estimation import (
    run_ekf_on_frame, load_ground_truth_dataset, load_imu_measurements,
    E3, P0_DIAG, ALPHA_DEFAULT, MEAS_STD_DEG_DEF, STEPS_DEFAULT, GAMMA_DEFAULT,
)
from compare_oracle_vs_ekf_wrench_estimation import (
    estimate_modal_oracle, run_ekf_for_order,
    wrench_from_modal, fwd_transform_general,
)

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
OUT_DIR   = ROOT / "scripts" / "force_est" / "debug_output" / "followup"
GT_PATH   = CRT / "gt_data" / "kirchhoff_gt_dataset.npz"
IMU2_PATH = CRT / "gt_data" / "kirchhoff_imu_2imu.npz"
IMU3_PATH = CRT / "gt_data" / "kirchhoff_imu_3imu.npz"

N_CASES = 3
N_NOISE = 5
_L      = 0.1

ORDERS = [
    (1, 1, 0),
    (2, 2, 0),
    (2, 2, 1),
    (3, 3, 2),
]
ORDER_LABELS = ["(1,1,0)", "(2,2,0)", "(2,2,1)", "(3,3,2)"]

METHOD_STYLES = {
    "GT":        dict(color="k",        lw=1.8, ls="-",  label="GT"),
    "Oracle":    dict(color="#2ca02c",  lw=1.4, ls="--", label="Oracle"),
    "EKF 2-IMU": dict(color="#1f77b4",  lw=1.2, ls="--", label="EKF 2-IMU"),
    "EKF 3-IMU": dict(color="#d62728",  lw=1.2, ls=":",  label="EKF 3-IMU"),
}
ARROW_COLORS = {
    "GT":        "k",
    "Oracle":    "#2ca02c",
    "EKF 2-IMU": "#1f77b4",
    "EKF 3-IMU": "#d62728",
}

# ---------------------------------------------------------------------------
# Build backbone from modal state (general order)
# ---------------------------------------------------------------------------
N_PTS_PLOT = 40   # arc-length points for backbone rendering

def backbone_from_modal(m, ox, oy, oz, n_pts=N_PTS_PLOT, L=_L, gamma=GAMMA_DEFAULT):
    s_arr = np.linspace(0.0, 1.0, n_pts)
    pos = np.zeros((n_pts, 3))
    for i, s in enumerate(s_arr):
        T = fwd_transform_general(m, float(s), gamma, L, ox, oy, oz)
        pos[i] = T[:3, 3]
    return pos   # (n_pts, 3) in metres


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_all():
    print("Loading datasets...")
    pos_gt, ori_gt, case_ids, gt_meta = load_ground_truth_dataset(GT_PATH)
    gt_np = np.load(GT_PATH, allow_pickle=True)
    tau   = gt_np["tau"]
    f_ext = gt_np["f_ext"]
    L     = float(gt_meta.get("length_m", _L))

    _, R_meas2, _, _, meta2 = load_imu_measurements(IMU2_PATH)
    _, R_meas3, _, _, meta3 = load_imu_measurements(IMU3_PATH)
    pos2 = np.array(meta2.get("imu_actual_s", [0.505, 1.0]), dtype=float)
    pos3 = np.array(meta3.get("imu_actual_s", [0.253, 0.505, 1.0]), dtype=float)

    # results[order_idx][case_idx][method] = {pos, force}
    results = [[{} for _ in range(N_CASES)] for _ in range(len(ORDERS))]

    for oi, (ox, oy, oz) in enumerate(ORDERS):
        lbl = ORDER_LABELS[oi]
        print(f"\n  Order {lbl} ...")
        for ci in range(N_CASES):
            # --- GT ---
            results[oi][ci]["GT"] = {
                "pos":   pos_gt[ci],          # (100, 3)
                "force": f_ext[ci],
            }

            # --- Oracle ---
            try:
                m_o = estimate_modal_oracle(pos_gt[ci], ori_gt[ci], ox, oy, oz, L)
                bb  = backbone_from_modal(m_o, ox, oy, oz, L=L)
                f_o, _ = wrench_from_modal(m_o, tau[ci], ox, oy, oz)
                results[oi][ci]["Oracle"] = {"pos": bb, "force": f_o}
            except Exception as e:
                print(f"    oracle ci={ci} failed: {e}")

            # --- EKF 2-IMU (mean over noise) ---
            ekf2_pos, ekf2_f = [], []
            for ni in range(N_NOISE):
                R_frame = [R_meas2[ci, ni, si] for si in range(len(pos2))]
                try:
                    m_e = run_ekf_for_order(
                        R_frame=R_frame, imu_pos_norm=pos2,
                        order_x=ox, order_y=oy, order_z=oz,
                        gamma=GAMMA_DEFAULT, meas_std_deg=MEAS_STD_DEG_DEF,
                        alpha=ALPHA_DEFAULT, steps=STEPS_DEFAULT,
                    )
                    ekf2_pos.append(backbone_from_modal(m_e, ox, oy, oz, L=L))
                    fe, _ = wrench_from_modal(m_e, tau[ci], ox, oy, oz)
                    ekf2_f.append(fe)
                except Exception:
                    pass
            if ekf2_pos:
                results[oi][ci]["EKF 2-IMU"] = {
                    "pos":   np.mean(ekf2_pos, axis=0),
                    "force": np.mean(ekf2_f, axis=0),
                }

            # --- EKF 3-IMU (mean over noise) ---
            ekf3_pos, ekf3_f = [], []
            for ni in range(N_NOISE):
                R_frame = [R_meas3[ci, ni, si] for si in range(len(pos3))]
                try:
                    m_e = run_ekf_for_order(
                        R_frame=R_frame, imu_pos_norm=pos3,
                        order_x=ox, order_y=oy, order_z=oz,
                        gamma=GAMMA_DEFAULT, meas_std_deg=MEAS_STD_DEG_DEF,
                        alpha=ALPHA_DEFAULT, steps=STEPS_DEFAULT,
                    )
                    ekf3_pos.append(backbone_from_modal(m_e, ox, oy, oz, L=L))
                    fe, _ = wrench_from_modal(m_e, tau[ci], ox, oy, oz)
                    ekf3_f.append(fe)
                except Exception:
                    pass
            if ekf3_pos:
                results[oi][ci]["EKF 3-IMU"] = {
                    "pos":   np.mean(ekf3_pos, axis=0),
                    "force": np.mean(ekf3_f, axis=0),
                }

            print(f"    case {ci}: done")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overlays(results, save_path_stem):
    n_orders = len(ORDERS)
    n_cases  = N_CASES
    methods  = ["GT", "Oracle", "EKF 2-IMU", "EKF 3-IMU"]

    fig = plt.figure(figsize=(4.5 * n_cases, 4.2 * n_orders))
    fig.suptitle("Shape Overlays with Tip Force Vectors\n"
                 "(black=GT, green=Oracle, blue=EKF 2-IMU, red=EKF 3-IMU)",
                 fontsize=11, y=0.99)

    # Determine global force scale: arrow length = force_scale * ||f||
    all_f_norms = []
    for oi in range(n_orders):
        for ci in range(n_cases):
            for m in methods:
                if m in results[oi][ci]:
                    fn = np.linalg.norm(results[oi][ci][m]["force"])
                    if fn > 0:
                        all_f_norms.append(fn)
    median_f = np.median(all_f_norms) if all_f_norms else 1.0
    arrow_scale = (_L * 0.35) / median_f   # arrows ~35% of rod length

    axes = []
    for oi in range(n_orders):
        row_axes = []
        for ci in range(n_cases):
            ax = fig.add_subplot(n_orders, n_cases,
                                 oi * n_cases + ci + 1, projection="3d")
            row_axes.append(ax)

            # Collect all positions to set consistent axis limits
            all_pos = []
            for m in methods:
                if m in results[oi][ci]:
                    all_pos.append(results[oi][ci][m]["pos"])
            if not all_pos:
                continue
            all_pos = np.vstack(all_pos)

            # Draw backbones
            for m in methods:
                if m not in results[oi][ci]:
                    continue
                p  = results[oi][ci][m]["pos"]
                st = METHOD_STYLES[m]
                ax.plot(p[:, 0], p[:, 1], p[:, 2],
                        color=st["color"], lw=st["lw"], ls=st["ls"],
                        label=st["label"] if ci == 0 else None)

            # Draw tip force arrows
            tip_gt = results[oi][ci]["GT"]["pos"][-1]
            for m in methods:
                if m not in results[oi][ci]:
                    continue
                f = results[oi][ci][m]["force"]
                fn = np.linalg.norm(f)
                if fn < 1e-10:
                    continue
                fhat = f / fn
                tip_m = results[oi][ci][m]["pos"][-1]
                ax.quiver(
                    tip_m[0], tip_m[1], tip_m[2],
                    fhat[0] * fn * arrow_scale,
                    fhat[1] * fn * arrow_scale,
                    fhat[2] * fn * arrow_scale,
                    color=ARROW_COLORS[m],
                    linewidth=1.6 if m == "GT" else 1.2,
                    arrow_length_ratio=0.25,
                    alpha=0.9,
                )

            # Mark base
            ax.scatter([0], [0], [0], color="k", s=18, zorder=5)

            # Axis formatting
            pad = _L * 0.08
            lo = all_pos.min(axis=0) - pad
            hi = all_pos.max(axis=0) + pad
            span = max((hi - lo).max(), _L * 0.25)
            mid  = (lo + hi) / 2
            for dim, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
                setter(mid[dim] - span / 2, mid[dim] + span / 2)

            ax.set_xlabel("x [m]", fontsize=7, labelpad=1)
            ax.set_ylabel("y [m]", fontsize=7, labelpad=1)
            ax.set_zlabel("z [m]", fontsize=7, labelpad=1)
            ax.tick_params(labelsize=6)
            ax.set_box_aspect([1, 1, 1])

            title = f"Order {ORDER_LABELS[oi]}  |  case {ci}"
            if oi == 0:
                title = f"Case {ci}\n" + title
            ax.set_title(title, fontsize=8, pad=3)

        axes.append(row_axes)

    # Legend (once, from first row first case)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=4,
               fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])

    for ext in ("pdf", "png"):
        p = Path(str(save_path_stem) + f".{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = compute_all()

    stem = OUT_DIR / "shape_wrench_overlays"
    print("\nRendering figure...")
    plot_overlays(results, stem)
    print("Done.")
