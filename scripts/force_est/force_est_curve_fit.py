import argparse
import json
import os
import sys
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

##############################################################################
# Example:
# python scripts\curve\3d_curve_fit_from_crt_gt.py \
#   --npz artifacts\data\tdcr_gt_fx_neg2_inext.npz \
#   --sample-idx 0 --order-x 1 --order-y 2 --order-z 0 \
#   --compute-wrench --plot-wrench-sample 0 --wrench-compare world_tip
##############################################################################

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from shape_force_est_imu.crt import virtual_work as vw


def forward_kinematics_multiple(m, s_values, gamma, L, order_x, order_y, order_z):
    rots = []
    poss = []
    for s in s_values:
        T_s = vw.product_of_exponentials(m, s, gamma, L, order_x, order_y, order_z)
        rots.append(T_s[:3, :3])
        poss.append(T_s[:3, 3])
    return rots, poss


##############################################################################
# Cost Function
##############################################################################
def orientation_error_angle(R_pred, R_obs):
    M = R_pred @ R_obs.T
    cos_val = (np.trace(M) - 1.0) / 2.0
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(np.arccos(cos_val))


def fit_cost(m, s_values, T_obs_list, gamma, L, w_pos, w_rot, order_x, order_y, order_z):
    cost = 0.0
    for s, T_obs in zip(s_values, T_obs_list):
        T_pred = vw.product_of_exponentials(m, s, gamma, L, order_x, order_y, order_z)
        dp = T_pred[:3, 3] - T_obs[:3, 3]
        ang = orientation_error_angle(T_pred[:3, :3], T_obs[:3, :3])
        cost += w_pos * float(dp @ dp) + w_rot * float(ang * ang)
    return float(cost)


##############################################################################
# Utilities
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


def parse_meta_dict(meta_raw):
    if meta_raw is None:
        return {}
    try:
        if isinstance(meta_raw, np.ndarray):
            meta_raw = meta_raw.item()
        meta = json.loads(meta_raw)
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def select_fit_indices(n_disks, disk_step):
    idx = list(range(0, n_disks, disk_step))
    if idx[-1] != n_disks - 1:
        idx.append(n_disks - 1)
    return np.array(idx, dtype=int)


def rotation_matrix_to_rotvec(R):
    cos_val = (np.trace(R) - 1.0) / 2.0
    cos_val = np.clip(cos_val, -1.0, 1.0)
    theta = float(np.arccos(cos_val))
    if theta < 1e-12:
        return np.zeros(3)
    w_hat = (R - R.T) / (2.0 * np.sin(theta))
    return theta * np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])


def estimate_modal_from_gt_frames(T_obs, s_values, order_x, order_y, order_z):
    s_mid = 0.5 * (s_values[:-1] + s_values[1:])
    ds = s_values[1] - s_values[0]
    kappa = np.zeros((len(s_mid), 3))
    for i in range(len(s_mid)):
        R_i = T_obs[i, :3, :3]
        R_j = T_obs[i + 1, :3, :3]
        R_rel = R_i.T @ R_j
        rotvec = rotation_matrix_to_rotvec(R_rel)
        kappa[i, :] = rotvec / ds

    def fit_axis(kappa_axis, order):
        Phi = np.vstack([s_mid ** i for i in range(order + 1)]).T
        return np.linalg.lstsq(Phi, kappa_axis, rcond=None)[0]

    mx = fit_axis(kappa[:, 0], order_x)
    my = fit_axis(kappa[:, 1], order_y)
    mz = fit_axis(kappa[:, 2], order_z)
    return np.hstack([mx, my, mz])


##############################################################################
# Wrench transforms
##############################################################################
def wrench_body_tip_to_world_tip(F_b6, T_tip):
    """Keep reference point at tip origin; rotate only."""
    R = T_tip[:3, :3]
    return np.hstack([R @ F_b6[:3], R @ F_b6[3:]])


def wrench_world_tip_to_body_tip(F_w6, T_tip):
    """Keep reference point at tip origin; rotate only."""
    R = T_tip[:3, :3]
    return np.hstack([R.T @ F_w6[:3], R.T @ F_w6[3:]])


def wrench_world_tip_to_world_base(F_w6_tip, p_tip):
    """Shift reference point from tip origin to base/world origin (same axes)."""
    M_tip = F_w6_tip[:3]
    F_tip = F_w6_tip[3:]
    M_base = M_tip + np.cross(p_tip, F_tip)
    return np.hstack([M_base, F_tip])


##############################################################################
# Main
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="tdcr_gt_samples_10.npz",
                        help="CRT GT npz file containing T, f_ext, l_ext, tau (and optional meta)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="number of samples to fit (default: all)")
    parser.add_argument("--sample-idx", type=int, default=None,
                        help="fit only this sample index (overrides n-samples)")
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
    parser.add_argument("--use-gt-curvature", action="store_true",
                        help="derive modal coefficients from GT frames instead of fitting")

    # Wrench estimation
    parser.add_argument("--compute-wrench", action="store_true",
                        help="estimate tip wrench using virtual work")
    parser.add_argument("--plot-wrench-sample", type=int, default=-1,
                        help="plot GT vs estimated wrench for a sample index")
    parser.add_argument("--plot-wrench-all", action="store_true",
                        help="plot GT vs estimated wrench for all computed samples")

    parser.add_argument("--tendon-offset", type=float, default=0.008,
                        help="tendon radial offset from backbone center (m)")
    parser.add_argument("--tendon-radius", type=float, default=None,
                        help="deprecated alias for --tendon-offset")
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

    # How to compare wrench vs GT
    # - world_tip: compare in WORLD axes, wrench about TIP origin (recommended for your GT)
    # - body_tip:  compare in BODY(tip) axes, wrench about TIP origin
    # - world_base: compare in WORLD axes, wrench about BASE origin (adds p×F to moments)
    parser.add_argument("--wrench-compare", type=str, default="world_tip",
                        choices=["world_tip", "body_tip", "world_base"],
                        help="How to express estimated wrench when comparing to GT")

    # sanity check that GT moment is tip-origin: expects l_ext ≈ 0 when pure force
    parser.add_argument("--print-pxF-check", action="store_true",
                        help="Print |l_ext - p×f_ext| to diagnose GT reference point (tip vs base)")
    parser.add_argument("--report-normalized-wrench", action="store_true",
                        help="report normalized wrench metrics using F0=EI/L^2, M0=EI/L")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="epsilon for normalized wrench metrics")
    parser.add_argument("--report-wrench-stats", action="store_true",
                        help="report mean±std for normalized metrics")

    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    T_all = data["T"]  # (N, n_disks, 4, 4)
    n_total, n_disks = T_all.shape[0], T_all.shape[1]

    if args.sample_idx is not None:
        if not (0 <= args.sample_idx < n_total):
            raise ValueError(f"sample-idx {args.sample_idx} out of range 0..{n_total-1}")
        sample_indices = [args.sample_idx]
    else:
        n_samples = n_total if args.n_samples is None else max(1, min(args.n_samples, n_total))
        sample_indices = list(range(n_samples))
    n_samples = len(sample_indices)

    meta = parse_meta_dict(data.get("meta", None))
    L_meta = float(meta.get("length_m", 0.0)) if meta else None
    L = args.length if args.length is not None else (L_meta if L_meta else 0.1)
    tendon_offset = args.tendon_offset
    if args.tendon_radius is not None:
        tendon_offset = args.tendon_radius
    if meta:
        gt_E = meta.get("youngs_modulus")
        gt_r = meta.get("backbone_radius")
        gt_tendon = meta.get("tendon_offset")
        if gt_E is not None and float(gt_E) != float(args.E):
            print(f"[warn] GT E={gt_E} but estimator E={args.E}")
        if gt_r is not None and float(gt_r) != float(args.backbone_radius):
            print(f"[warn] GT backbone_radius={gt_r} but estimator backbone_radius={args.backbone_radius}")
        if gt_tendon is not None and float(gt_tendon) != float(tendon_offset):
            print(f"[warn] GT tendon_offset={gt_tendon} but estimator tendon_offset={tendon_offset}")

    order_x, order_y, order_z = args.order_x, args.order_y, args.order_z
    n_params = vw.total_params(order_x, order_y, order_z)

    s_values = np.linspace(0.0, 1.0, n_disks)
    fit_idx = select_fit_indices(n_disks, args.disk_step)
    s_fit = s_values[fit_idx]

    pos_err_sum = np.zeros(n_disks)
    rot_err_sum = np.zeros(n_disks)

    m_prev = np.zeros(n_params)
    est_list = []

    # ------------------------------------------------------------------
    # Fit shape states
    # ------------------------------------------------------------------
    total_samples = len(sample_indices)
    for idx, si in enumerate(sample_indices, start=1):
        print(f"[fit] sample {idx}/{total_samples} (index={si})")
        T_obs = T_all[si]
        if args.use_gt_curvature:
            m_est = estimate_modal_from_gt_frames(
                T_obs, s_values, order_x, order_y, order_z
            )
        else:
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
        est_list.append((si, m_est))
        m_prev = m_est

        rots_est, pos_est = forward_kinematics_multiple(
            m_est, s_values, args.gamma, L, order_x, order_y, order_z
        )
        for j in range(n_disks):
            dp = pos_est[j] - T_obs[j, :3, 3]
            pos_err_sum[j] += float(np.linalg.norm(dp))
            rot_err_sum[j] += float(orientation_error_angle(rots_est[j], T_obs[j, :3, :3]))

    pos_err_avg = pos_err_sum / n_samples
    rot_err_avg = rot_err_sum / n_samples

    print(f"Fitted samples: {n_samples}")
    print("Average position error per disk (m):")
    print(pos_err_avg)
    print("Average rotation error per disk (rad):")
    print(rot_err_avg)

    # ------------------------------------------------------------------
    # Wrench estimation
    # ------------------------------------------------------------------
    if args.compute_wrench:
        tau_all = data["tau"]
        f_ext_all = data["f_ext"]
        l_ext_all = data["l_ext"]

        tendon_count = tau_all.shape[1]
        angles = np.linspace(0.0, 2.0 * np.pi, tendon_count, endpoint=False)
        r_list = [
            np.array([tendon_offset * np.cos(a), tendon_offset * np.sin(a), 0.0])
            for a in angles
        ]

        # Elastic parameters
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
        est_wrenches_cache = []  # (idx, F_est_cmp6, F_gt_cmp6, T_tip)

        for si, m_est in est_list:
            # Jacobians & tip transform from estimated shape
            J_vb_m, T_tip = vw.body_jacobian_at_s(
                m_est, 1.0, args.gamma, L, order_x, order_y, order_z
            )
            J_lm = vw.cable_jacobian(m_est, r_list, L, order_x, order_y, order_z)
            gradU = vw.elastic_energy_gradient(m_est, EIx, EIy, GJ, L, order_x, order_y, order_z)

            # Solve BODY(tip) wrench (this is about tip origin)
            F_b = vw.solve_wrench(J_vb_m, J_lm, gradU, tau_all[si], args.pinv_rcond)

            # GT world-tip wrench (your dataset convention)
            F_gt_world_tip = np.hstack([l_ext_all[si], f_ext_all[si]])

            # Convert estimate to desired compare form
            if args.wrench_compare == "world_tip":
                F_est_cmp = wrench_body_tip_to_world_tip(F_b, T_tip)
                F_gt_cmp = F_gt_world_tip
            elif args.wrench_compare == "body_tip":
                F_est_cmp = F_b
                F_gt_cmp = wrench_world_tip_to_body_tip(F_gt_world_tip, T_tip)
            else:  # world_base
                # shift GT and EST from tip-origin to base-origin consistently
                p_tip = T_tip[:3, 3]
                F_est_world_tip = wrench_body_tip_to_world_tip(F_b, T_tip)
                F_est_cmp = wrench_world_tip_to_world_base(F_est_world_tip, p_tip)
                F_gt_cmp = wrench_world_tip_to_world_base(F_gt_world_tip, p_tip)

            moment_err.append(float(np.linalg.norm(F_est_cmp[:3] - F_gt_cmp[:3])))
            force_err.append(float(np.linalg.norm(F_est_cmp[3:] - F_gt_cmp[3:])))

            est_wrenches_cache.append((si, F_est_cmp, F_gt_cmp, T_tip))

            if args.print_pxF_check:
                # If GT is tip-origin, l_ext should NOT match p×f_ext (base-origin moment),
                # so |l_ext - p×f_ext| will be ~|p×f_ext|, not ~0.
                p_gt = T_all[si, -1, :3, 3]
                pxF = np.cross(p_gt, f_ext_all[si])
                diff = np.linalg.norm(l_ext_all[si] - pxF)
                print(f"[pxF-check] sample {si}: |l_ext - p×f_ext| = {diff:.6g}   |p×f|={np.linalg.norm(pxF):.6g}")

        print("Wrench compare mode:", args.wrench_compare)
        print("Average tip force error (N):", float(np.mean(force_err)))
        print("Average tip moment error (Nm):", float(np.mean(moment_err)))

        if args.report_normalized_wrench:
            I = np.pi * (args.backbone_radius ** 4) / 4.0
            EI = args.E * I
            F0 = EI / (L ** 2)
            M0 = EI / L
            eps = args.eps
            f_errs = []
            m_errs = []
            f_mag_errs = []
            m_mag_errs = []
            f_dir_errs = []
            m_dir_errs = []
            for _, Fest, Fgt, _ in est_wrenches_cache:
                f_est = Fest[3:]
                f_gt = Fgt[3:]
                m_est = Fest[:3]
                m_gt = Fgt[:3]
                f_est_n = f_est / F0
                f_gt_n = f_gt / F0
                m_est_n = m_est / M0
                m_gt_n = m_gt / M0
                f_errs.append(float(np.linalg.norm(f_est_n - f_gt_n)))
                m_errs.append(float(np.linalg.norm(m_est_n - m_gt_n)))
                f_mag_errs.append(float(np.linalg.norm(f_est - f_gt) / (np.linalg.norm(f_gt) + eps)))
                m_mag_errs.append(float(np.linalg.norm(m_est - m_gt) / (np.linalg.norm(m_gt) + eps)))
                f_cos = float(np.dot(f_est, f_gt) / (np.linalg.norm(f_est) * np.linalg.norm(f_gt) + eps))
                m_cos = float(np.dot(m_est, m_gt) / (np.linalg.norm(m_est) * np.linalg.norm(m_gt) + eps))
                f_dir_errs.append(float(np.degrees(np.arccos(np.clip(f_cos, -1.0, 1.0)))))
                m_dir_errs.append(float(np.degrees(np.arccos(np.clip(m_cos, -1.0, 1.0)))))
            nrmse_f = float(np.sqrt(np.mean(np.square(f_errs))))
            nrmse_m = float(np.sqrt(np.mean(np.square(m_errs))))
            print(f"NRMSE_f (normalized): {nrmse_f}")
            print(f"NRMSE_tau (normalized): {nrmse_m}")
            print("Relative magnitude error f (avg):", float(np.mean(f_mag_errs)))
            print("Relative magnitude error tau (avg):", float(np.mean(m_mag_errs)))
            print("Direction error f (deg, avg):", float(np.mean(f_dir_errs)))
            print("Direction error tau (deg, avg):", float(np.mean(m_dir_errs)))
            if args.report_wrench_stats:
                def mean_std(vals):
                    return float(np.mean(vals)), float(np.std(vals))
                f_err_mu, f_err_sd = mean_std(f_errs)
                m_err_mu, m_err_sd = mean_std(m_errs)
                f_mag_mu, f_mag_sd = mean_std(f_mag_errs)
                m_mag_mu, m_mag_sd = mean_std(m_mag_errs)
                f_dir_mu, f_dir_sd = mean_std(f_dir_errs)
                m_dir_mu, m_dir_sd = mean_std(m_dir_errs)
                print("Norm err f (normalized) mean±std:", f_err_mu, f_err_sd)
                print("Norm err tau (normalized) mean±std:", m_err_mu, m_err_sd)
                print("Rel mag err f mean±std:", f_mag_mu, f_mag_sd)
                print("Rel mag err tau mean±std:", m_mag_mu, m_mag_sd)
                print("Dir err f (deg) mean±std:", f_dir_mu, f_dir_sd)
                print("Dir err tau (deg) mean±std:", m_dir_mu, m_dir_sd)

        # Plot one sample wrench
        if 0 <= args.plot_wrench_sample < n_total:
            sample_idx = args.plot_wrench_sample
            cache_map = {idx: (Fest, Fgt, Ttip) for idx, Fest, Fgt, Ttip in est_wrenches_cache}
            if sample_idx not in cache_map:
                raise ValueError("Requested wrench plot sample not computed.")
            Fest, Fgt, Ttip = cache_map[sample_idx]

            labels = ["Mx", "My", "Mz", "Fx", "Fy", "Fz"]

            fig, axes = plt.subplots(2, 3, figsize=(9, 5))
            for k, ax in enumerate(axes.ravel()):
                ax.bar([0], [Fgt[k]], width=0.4, color="blue", label="GT")
                ax.bar([1], [Fest[k]], width=0.4, color="green", label="Est")
                ax.set_title(labels[k])
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["GT", "Est"])
                ax.grid(True, axis="y", alpha=0.3)

            handles, labels_ = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels_, loc="upper right")
            fig.suptitle(f"Wrench components (sample {sample_idx}) mode={args.wrench_compare}")
            plt.tight_layout()
            plt.show()

        if args.plot_wrench_all:
            for sample_idx, Fest, Fgt, _ in est_wrenches_cache:
                labels = ["Mx", "My", "Mz", "Fx", "Fy", "Fz"]

                fig, axes = plt.subplots(2, 3, figsize=(9, 5))
                for k, ax in enumerate(axes.ravel()):
                    ax.bar([0], [Fgt[k]], width=0.4, color="blue", label="GT")
                    ax.bar([1], [Fest[k]], width=0.4, color="green", label="Est")
                    ax.set_title(labels[k])
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(["GT", "Est"])
                    ax.grid(True, axis="y", alpha=0.3)

                handles, labels_ = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels_, loc="upper right")
                fig.suptitle(f"Wrench components (sample {sample_idx}) mode={args.wrench_compare}")
                plt.tight_layout()
                plt.show()

    # ------------------------------------------------------------------
    # Plot shape
    # ------------------------------------------------------------------
    def plot_sample(sample_idx, m_plot):
        T_plot = T_all[sample_idx]
        _, pos_est = forward_kinematics_multiple(
            m_plot, s_values, args.gamma, L, order_x, order_y, order_z
        )
        pos_est = np.array(pos_est)
        pos_gt = T_plot[:, :3, 3]
        all_pts = np.vstack([pos_gt, pos_est])
        mins = np.min(all_pts, axis=0)
        maxs = np.max(all_pts, axis=0)
        centers = 0.5 * (mins + maxs)
        max_range = float(np.max(maxs - mins))
        if max_range == 0.0:
            max_range = 1.0

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2],
                color="blue", label="GT shape")
        ax.plot(pos_est[:, 0], pos_est[:, 1], pos_est[:, 2],
                color="green", linestyle="--", label="Estimated shape")
        ax.set_xlim(centers[0] - 0.5 * max_range, centers[0] + 0.5 * max_range)
        ax.set_ylim(centers[1] - 0.5 * max_range, centers[1] + 0.5 * max_range)
        ax.set_zlim(centers[2] - 0.5 * max_range, centers[2] + 0.5 * max_range)
        try:
            ax.set_box_aspect([1.0, 1.0, 1.0])
        except AttributeError:
            pass
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title(f"Sample {sample_idx}")
        ax.legend()
        plt.tight_layout()
        plt.show()

    if args.plot_all:
        for sample_idx, m_plot in est_list:
            plot_sample(sample_idx, m_plot)
    elif 0 <= args.plot_sample < n_total:
        est_map = {idx: m for idx, m in est_list}
        if args.plot_sample in est_map:
            plot_sample(args.plot_sample, est_map[args.plot_sample])
        elif args.sample_idx is not None:
            raise ValueError("Requested plot sample not computed.")


if __name__ == "__main__":
    main()
