# crt_time_comparison.py
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from cosserat_rod_model import CosseratRodModel
from pck_models import pck0_inverse_2d_ls_value, pck2_inverse_2d_ls_value, compute_external_wrench

# -------------------------------------------------------------
# Simplified in-plane Cosserat rod: no shear, no torsion
# -------------------------------------------------------------
def simplified_crt_forward(tau, length=0.04, num_disks=20):
    """
    Forward integration for in-plane bending only, no shear/extension/torsion.
    Assumes constant curvature equivalent to a uniform moment.
    """
    EI = 1.0  # normalized bending stiffness (unitless for comparison)
    M_net = np.sum(tau) * 0.01  # pretend a simple net bending moment from all tendons
    kappa = M_net / EI          # constant curvature

    def ode_fun(s, y):
        x, z, theta = y
        return [math.sin(theta), math.cos(theta), kappa]

    y0 = [0.0, 0.0, 0.0]
    s_vals = np.linspace(0, length, num_disks+1)
    sol = solve_ivp(ode_fun, (0, length), y0, t_eval=s_vals, rtol=1e-6, atol=1e-8)
    return sol.t, sol.y

# -------------------------------------------------------------
# Utility: Estimate external wrench by fitting CRT model to tip pose
# -------------------------------------------------------------
def estimate_external_wrench_from_tip_pose(tip_pose_target, tau,
                                           length=0.04, r_anchor=0.0018, E=6.5e10,
                                           num_disks=20):
    def residual(F_ext_guess):
        f_ext = F_ext_guess[:3]
        l_ext = F_ext_guess[3:]
        model = CosseratRodModel(length=length,
                                  backbone_radius=r_anchor,
                                  youngs_modulus=E,
                                  num_disks=num_disks)
        try:
            T_tip = model.forward_kinematics(tau, f_ext=f_ext, l_ext=l_ext)
        except RuntimeError:
            return 1e3 * np.ones(6)

        T_err = np.zeros(6)
        T_est = T_tip[:3, 3]
        T_tar = tip_pose_target[:3, 3]
        T_err[:3] = T_est - T_tar

        R_est = T_tip[:3, :3]
        R_tar = tip_pose_target[:3, :3]
        R_delta = R_tar.T @ R_est
        angle = math.acos(np.clip((np.trace(R_delta) - 1)/2.0, -1.0, 1.0))
        T_err[3:] = angle * 0.1 * np.ones(3)

        return T_err

    x0 = np.array([0.0, 0.0, -0.02, 0.0, 0.001, 0.0])
    sol = least_squares(residual, x0, method='lm', xtol=1e-6, ftol=1e-6, max_nfev=100)
    return sol.x

# -------------------------------------------------------------
# Compare PCK0, PCK2 (virtual work) and CRT (direct) timing
# -------------------------------------------------------------
if __name__ == '__main__':
    L = 0.04
    r = 0.0018
    E = 6.5e10
    I = 4.83e-15
    delta_deg = 45.0
    delta = math.radians(delta_deg)
    tau = np.array([2.17, 2.05, 0.0, 0.0])
    q1 = 0.002

    disks_list = list(range(1, 21))
    N_repeat = 1

    shape_pck0, shape_pck2, shape_crt = [], [], []
    load_pck0, load_pck2, load_crt = [], [], []
    std_shape_pck0, std_shape_pck2, std_shape_crt = [], [], []
    std_load_pck0, std_load_pck2, std_load_crt = [], [], []

    for nd in disks_list:
        t_s_pck0, t_s_pck2, t_s_crt = [], [], []
        t_l_pck0, t_l_pck2, t_l_crt = [], [], []

        for _ in range(N_repeat):
            # PCK0 shape only
            t0 = time.perf_counter()
            m0 = pck0_inverse_2d_ls_value(0.02, 0.03, math.radians(30), L=L)
            t1 = time.perf_counter()
            t_s_pck0.append((t1 - t0)*1e3)

            # PCK0 shape + load
            t0 = time.perf_counter()
            compute_external_wrench(m0, 0.0, 0.0, delta, E, I, L, r, q1, tau, theta_meas=m0)
            t1 = time.perf_counter()
            t_l_pck0.append((t1 - t0)*1e3)

            # PCK2 shape only
            t0 = time.perf_counter()
            m0, m1, m2 = pck2_inverse_2d_ls_value(0.02, 0.03, math.radians(30), L=L)
            t1 = time.perf_counter()
            t_s_pck2.append((t1 - t0)*1e3)

            # PCK2 shape + load
            t0 = time.perf_counter()
            compute_external_wrench(m0, m1, m2, delta, E, I, L, r, q1, tau, theta_meas=m0)
            t1 = time.perf_counter()
            t_l_pck2.append((t1 - t0)*1e3)

            # CRT shape only
            model = CosseratRodModel(length=L, backbone_radius=r, youngs_modulus=E, num_disks=nd)
            t0 = time.perf_counter()
            T_tip_gt, _ = model.forward_kinematics(tau, f_ext=np.array([0.0, 0.0, -0.02]),
                                                   l_ext=np.array([0.0, 0.001, 0.0]), return_states=True)
            t1 = time.perf_counter()
            t_s_crt.append((t1 - t0)*1e3)

            # CRT shape + external wrench estimation
            t0 = time.perf_counter()
            estimate_external_wrench_from_tip_pose(T_tip_gt, tau, length=L, r_anchor=r, E=E, num_disks=nd)
            t1 = time.perf_counter()
            t_l_crt.append((t1 - t0)*1e3)

        shape_pck0.append(np.mean(t_s_pck0))
        std_shape_pck0.append(np.std(t_s_pck0))
        shape_pck2.append(np.mean(t_s_pck2))
        std_shape_pck2.append(np.std(t_s_pck2))
        shape_crt.append(np.mean(t_s_crt))
        std_shape_crt.append(np.std(t_s_crt))

        load_pck0.append(np.mean(t_l_pck0))
        std_load_pck0.append(np.std(t_l_pck0))
        load_pck2.append(np.mean(t_l_pck2))
        std_load_pck2.append(np.std(t_l_pck2))
        load_crt.append(np.mean(t_l_crt))
        std_load_crt.append(np.std(t_l_crt))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Shape only times
    ax1.errorbar(disks_list, shape_pck0, yerr=std_shape_pck0, fmt='-o', label='PCK0 (CC)')
    ax1.errorbar(disks_list, shape_pck2, yerr=std_shape_pck2, fmt='-s', label='PCK2')
    ax1.errorbar(disks_list, shape_crt, yerr=std_shape_crt, fmt='-^', label='CRT')
    ax1.set_title("Computation Time: Shape Only")
    ax1.set_xlabel("Number of Disks")
    ax1.set_ylabel("Time (ms)")
    ax1.grid(True)
    ax1.legend()

    # Shape + Load
    ax2.errorbar(disks_list, load_pck0, yerr=std_load_pck0, fmt='-o', label='PCK0 (CC) + Load')
    ax2.errorbar(disks_list, load_pck2, yerr=std_load_pck2, fmt='-s', label='PCK2 + Load')
    ax2.errorbar(disks_list, load_crt, yerr=std_load_crt, fmt='-^', label='CRT (Inverse Wrench)')
    ax2.set_title("Computation Time: Shape + External Load")
    ax2.set_xlabel("Number of Disks")
    ax2.set_ylabel("Time (ms)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

