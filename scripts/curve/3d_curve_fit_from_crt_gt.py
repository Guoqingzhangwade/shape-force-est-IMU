import argparse
import json
import os
import sys
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from shape_force_est_imu.crt import virtual_work as vw

def forward_kinematics_multiple(m, s_values, gamma, L, order_x, order_y, order_z):
    rots = []
    poss = []
    for s in s_values:
        T_s = vw.product_of_exponentials(
            m, s, gamma, L, order_x, order_y, order_z
        )
        rots.append(T_s[:3, :3])
        poss.append(T_s[:3, 3])
    return rots, poss

##############################################################################
# 3) Cost Function
##############################################################################

def orientation_error_angle(R_pred, R_obs):
    M = R_pred @ R_obs.T
    cos_val = (np.trace(M) - 1.0) / 2.0
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return np.arccos(cos_val)

def fit_cost(m, s_values, T_obs_list, gamma, L, w_pos, w_rot,
             order_x, order_y, order_z):
    cost = 0.0
    for s, T_obs in zip(s_values, T_obs_list):
        T_pred = vw.product_of_exponentials(
            m, s, gamma, L, order_x, order_y, order_z
        )
        dp = T_pred[:3, 3] - T_obs[:3, 3]
        ang = orientation_error_angle(T_pred[:3, :3], T_obs[:3, :3])
        cost += w_pos * float(dp @ dp) + w_rot * float(ang * ang)
    return cost

##############################################################################
# 4) Utilities
##############################################################################

def parse_length_from_meta(meta_raw):
    if meta_raw is None:
        return None
    try:
        if isinstance(meta_raw, np.ndarray):
            meta_raw = meta_raw.item()
        meta = json.loads(meta_raw)
        return float(meta.get("length_m", 0.0)) if meta else None
    except Exception:
        return None

def select_fit_indices(n_disks, disk_step):
    idx = list(range(0, n_disks, disk_step))
    if idx[-1] != n_disks - 1:
        idx.append(n_disks - 1)
    return np.array(idx, dtype=int)

##############################################################################
# 5) Main
##############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="tdcr_gt_samples_10.npz",
                        help="CRT GT npz file containing T, f_ext, l_ext")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="number of samples to fit (default: all)")
    parser.add_argument("--gamma", type=int, default=26,
                        help="number of Magnus subintervals")
    parser.add_argument("--order-x", type=int, default=1,
                        help="polynomial order for kappa_x(s)")
    parser.add_argument("--order-y", type=int, default=1,
                        help="polynomial order for kappa_y(s)")
    parser.add_argument("--order-z", type=int, default=0,
                        help="polynomial order for kappa_z(s)")
    parser.add_argument("--disk-step", type=int, default=1,
                        help="subsample disks for fitting (1 = use all)")
    parser.add_argument("--w-pos", type=float, default=1.0,
                        help="position error weight")
    parser.add_argument("--w-rot", type=float, default=1.0,
                        help="rotation error weight")
    parser.add_argument("--plot-sample", type=int, default=0,
                        help="sample index to plot")
    parser.add_argument("--plot-all", action="store_true",
                        help="plot all samples sequentially")
    parser.add_argument("--length", type=float, default=None,
                        help="override backbone length in meters")
    parser.add_argument("--compute-wrench", action="store_true",
                        help="estimate tip wrench using virtual work")
    parser.add_argument("--plot-wrench-sample", type=int, default=-1,
                        help="plot GT vs estimated wrench for a sample index")
    parser.add_argument("--force-scale", type=float, default=0.02,
                        help="scale factor for force quiver length")
    parser.add_argument("--moment-scale", type=float, default=0.5,
                        help="scale factor for moment quiver length")
    parser.add_argument("--tendon-radius", type=float, default=0.008,
                        help="tendon routing radius in meters")
    parser.add_argument("--pinv-rcond", type=float, default=1e-8,
                        help="rcond for pseudoinverse in wrench solve")
    parser.add_argument("--E", type=float, default=60e9,
                        help="Young's modulus for elastic gradient")
    parser.add_argument("--poisson", type=float, default=0.3,
                        help="Poisson ratio for shear modulus")
    parser.add_argument("--backbone-radius", type=float, default=5e-4,
                        help="backbone radius in meters")
    parser.add_argument("--EIx", type=float, default=None,
                        help="override EI_x (N*m^2)")
    parser.add_argument("--EIy", type=float, default=None,
                        help="override EI_y (N*m^2)")
    parser.add_argument("--GJ", type=float, default=None,
                        help="override GJ (N*m^2)")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    T_all = data["T"]  # (N, n_disks, 4, 4)
    n_total, n_disks = T_all.shape[0], T_all.shape[1]
    n_samples = n_total if args.n_samples is None else min(args.n_samples, n_total)

    L_meta = parse_length_from_meta(data.get("meta", None))
    L = args.length if args.length is not None else (L_meta if L_meta else 0.1)

    order_x, order_y, order_z = args.order_x, args.order_y, args.order_z
    n_params = vw.total_params(order_x, order_y, order_z)
    s_values = np.linspace(0.0, 1.0, n_disks)
    fit_idx = select_fit_indices(n_disks, args.disk_step)
    s_fit = s_values[fit_idx]

    pos_err_sum = np.zeros(n_disks)
    rot_err_sum = np.zeros(n_disks)

    m_prev = np.zeros(n_params)
    est_list = []

    for i in range(n_samples):
        T_obs = T_all[i]
        T_fit = T_obs[fit_idx]

        res = minimize(
            fit_cost,
            m_prev,
            args=(s_fit, T_fit, args.gamma, L, args.w_pos, args.w_rot,
                  order_x, order_y, order_z),
            method="L-BFGS-B",
            options={"maxiter": 200}
        )
        m_est = res.x
        est_list.append(m_est)
        m_prev = m_est

        rots_est, pos_est = forward_kinematics_multiple(
            m_est, s_values, args.gamma, L, order_x, order_y, order_z
        )
        for j in range(n_disks):
            dp = pos_est[j] - T_obs[j, :3, 3]
            pos_err_sum[j] += np.linalg.norm(dp)
            rot_err_sum[j] += orientation_error_angle(rots_est[j], T_obs[j, :3, :3])

    pos_err_avg = pos_err_sum / n_samples
    rot_err_avg = rot_err_sum / n_samples

    print(f"Fitted samples: {n_samples}")
    print("Average position error per disk (m):")
    print(pos_err_avg)
    print("Average rotation error per disk (rad):")
    print(rot_err_avg)

    if args.compute_wrench:
        tau_all = data["tau"]
        f_ext_all = data["f_ext"]
        l_ext_all = data["l_ext"]
        tendon_count = tau_all.shape[1]
        angles = np.linspace(0.0, 2.0 * np.pi, tendon_count, endpoint=False)
        r_list = [
            np.array([args.tendon_radius * np.cos(a), args.tendon_radius * np.sin(a), 0.0])
            for a in angles
        ]
        if args.EIx is None or args.EIy is None or args.GJ is None:
            I = np.pi * (args.backbone_radius ** 4) / 4.0
            G = args.E / (2.0 * (1.0 + args.poisson))
            EIx = args.E * I if args.EIx is None else args.EIx
            EIy = args.E * I if args.EIy is None else args.EIy
            GJ = 2.0 * G * I if args.GJ is None else args.GJ
        else:
            EIx, EIy, GJ = args.EIx, args.EIy, args.GJ

        force_err = []
        moment_err = []
        est_wrenches = []
        for i in range(n_samples):
            m_est = est_list[i]
            J_vb_m, T_tip = vw.body_jacobian_at_s(
                m_est, 1.0, args.gamma, L, order_x, order_y, order_z
            )
            J_lm = vw.cable_jacobian(
                m_est, r_list, L, order_x, order_y, order_z
            )
            gradU = vw.elastic_energy_gradient(
                m_est, EIx, EIy, GJ, L, order_x, order_y, order_z
            )
            F_b = vw.solve_wrench(J_vb_m, J_lm, gradU, tau_all[i], args.pinv_rcond)
            Ad = vw.adjoint(T_tip)
            F_w = np.linalg.solve(Ad.T, F_b)
            est_wrenches.append(F_w)
            moment_err.append(np.linalg.norm(F_w[:3] - l_ext_all[i]))
            force_err.append(np.linalg.norm(F_w[3:] - f_ext_all[i]))

        print("Average tip force error (N):", float(np.mean(force_err)))
        print("Average tip moment error (Nm):", float(np.mean(moment_err)))

        if 0 <= args.plot_wrench_sample < n_samples:
            sample_idx = args.plot_wrench_sample
            F_est = est_wrenches[sample_idx][3:]
            M_est = est_wrenches[sample_idx][:3]
            F_gt = f_ext_all[sample_idx]
            M_gt = l_ext_all[sample_idx]

            labels = ["Mx", "My", "Mz", "Fx", "Fy", "Fz"]
            gt_vals = np.hstack([M_gt, F_gt])
            est_vals = np.hstack([M_est, F_est])

            fig, axes = plt.subplots(2, 3, figsize=(9, 5))
            for i, ax in enumerate(axes.ravel()):
                ax.bar([0], [gt_vals[i]], width=0.4, color="blue", label="GT")
                ax.bar([1], [est_vals[i]], width=0.4, color="green", label="Est")
                ax.set_title(labels[i])
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["GT", "Est"])
                ax.grid(True, axis="y", alpha=0.3)
            handles, labels_ = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels_, loc="upper right")
            fig.suptitle(f"Tip wrench components (sample {sample_idx})")
            plt.tight_layout()
            plt.show()

    def plot_sample(sample_idx):
        T_plot = T_all[sample_idx]
        m_plot = est_list[sample_idx]
        _, pos_est = forward_kinematics_multiple(
            m_plot, s_values, args.gamma, L, order_x, order_y, order_z
        )
        pos_est = np.array(pos_est)
        pos_gt = T_plot[:, :3, 3]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2],
                color="blue", label="GT shape")
        ax.plot(pos_est[:, 0], pos_est[:, 1], pos_est[:, 2],
                color="green", linestyle="--", label="Estimated shape")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title(f"Sample {sample_idx}")
        ax.legend()
        plt.tight_layout()
        plt.show()

    if args.plot_all:
        for sample_idx in range(n_samples):
            plot_sample(sample_idx)
    elif 0 <= args.plot_sample < n_samples:
        plot_sample(args.plot_sample)

if __name__ == "__main__":
    main()
