# crt_time_comparison.py
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from cosserat_rod_model import CosseratRodModel

# -------------------------------------------------------------
# Utility to compute force estimation using virtual work model (for PCK0, PCK2)
# -------------------------------------------------------------
def compute_external_wrench(m0, m1, m2, delta, E, I, L, r, q1, tau, theta_meas):
    def kappa(s):
        return m0 + m1*s + m2*(s**2)

    def theta_of_s(u):
        from scipy.integrate import quad
        val, _ = quad(kappa, 0, u)
        return val

    def Rz_negdelta(v):
        c = math.cos(-delta)
        s = math.sin(-delta)
        R = np.array([[ c, -s,  0], [ s,  c,  0], [ 0,  0,  1]])
        return R.dot(v)

    def partial_mi(s, i):
        x = math.cos(theta_of_s(s))*(s**(i+1)/(i+1))
        y = 0.0
        z = -math.sin(theta_of_s(s))*(s**(i+1)/(i+1))
        return np.array([x, y, z])

    def partial_delta(s):
        x = math.cos(theta_of_s(s))
        y = -math.sin(theta_of_s(s))
        z = 0.0
        return np.array([x, y, z])

    def integral_vec(func, a, b, N=100):
        s_vals = np.linspace(a, b, N+1)
        h = (b - a)/N
        out_vec = np.zeros_like(func(a))
        for i, s in enumerate(s_vals):
            w = 0.5 if (i==0 or i==N) else 1.0
            out_vec += w * func(s)
        return h*out_vec

    theta_e_calc = m0 + 0.5*m1 + (m2/3.0)
    vec_m0 = L * integral_vec(lambda ss: partial_mi(ss, 0), 0, 1)
    vec_m1 = L * integral_vec(lambda ss: partial_mi(ss, 1), 0, 1)
    vec_m2 = L * integral_vec(lambda ss: partial_mi(ss, 2), 0, 1)
    vec_delta = L * integral_vec(partial_delta, 0, 1)
    JpS = np.column_stack([Rz_negdelta(vec_m0),
                           Rz_negdelta(vec_m1),
                           Rz_negdelta(vec_m2),
                           Rz_negdelta(vec_delta)])

    c_th = math.cos(theta_e_calc)
    s_th = math.sin(theta_e_calc)
    c_dl = math.cos(delta)
    s_dl = math.sin(delta)
    JwS = np.array([
        [-c_th*s_dl, -0.5*c_th*s_dl, -(1.0/3.0)*c_th*s_dl, s_th*c_dl],
        [ c_th*c_dl,  0.5*c_th*c_dl,  (1.0/3.0)*c_th*c_dl, s_th*s_dl],
        [   -s_th,     -0.5*s_th,     -(1.0/3.0)*s_th,      (c_th-1.0)]
    ])
    JxS = np.vstack([JpS, JwS])

    beta = math.pi/2
    sigma_all = [0.0 + i*beta for i in range(4)]
    JqS = np.zeros((4,4))
    for iRow in range(4):
        sig_i = sigma_all[iRow]
        JqS[iRow,:] = -r*np.array([
            math.cos(sig_i),
            0.5*math.cos(sig_i),
            (1.0/3.0)*math.cos(sig_i),
            -theta_e_calc*math.sin(sig_i)
        ])

    gradU_S = L*np.array([
        E*I*(m0 + 0.25*m1 + (1.0/3.0)*m2),
        E*I*(0.25*m0 + (1.0/3.0)*m1 + 0.125*m2),
        E*I*((1.0/3.0)*m0 + 0.125*m1 + 0.2*m2),
        0.0
    ])

    diffVec_4 = gradU_S - JqS.T.dot(tau)
    F_est = np.linalg.pinv(JxS.T).dot(diffVec_4)
    return F_est

# -------------------------------------------------------------
# Wrapper for Cosserat Rod CRT Model
# -------------------------------------------------------------
def crt_inverse_static_3d(tau, length=0.04, r_anchor=0.0018, E=6.5e10,
                          num_disks=10, return_traj=False):
    model = CosseratRodModel(length=length, backbone_radius=r_anchor,
                              youngs_modulus=E, num_disks=num_disks)
    return model.forward_kinematics(tau, return_states=return_traj)

# -------------------------------------------------------------
# Test and Compare Timing
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
    N_repeat = 10

    time_pck0, time_pck2, time_crt = [], [], []
    std_pck0, std_pck2, std_crt = [], [], []

    for nd in disks_list:
        t_pck0_list, t_pck2_list, t_crt_list = [], [], []
        for _ in range(N_repeat):
            # Simulate PCK0
            t0 = time.perf_counter()
            m0 = 5.0  # example curvature guess
            F_pck0 = compute_external_wrench(m0, 0.0, 0.0,
                                             delta, E, I, L, r,
                                             q1, tau, theta_meas=m0)
            t1 = time.perf_counter()
            t_pck0_list.append((t1 - t0)*1e3)

            # Simulate PCK2
            t0 = time.perf_counter()
            m0, m1, m2 = 5.0, 0.0, 0.0
            F_pck2 = compute_external_wrench(m0, m1, m2,
                                             delta, E, I, L, r,
                                             q1, tau, theta_meas=m0)
            t1 = time.perf_counter()
            t_pck2_list.append((t1 - t0)*1e3)

            # Simulate CRT
            t0 = time.perf_counter()
            T_tip, traj = crt_inverse_static_3d(tau, length=L, r_anchor=r,
                                                E=E, num_disks=nd,
                                                return_traj=True)
            t1 = time.perf_counter()
            t_crt_list.append((t1 - t0)*1e3)

        time_pck0.append(np.mean(t_pck0_list))
        std_pck0.append(np.std(t_pck0_list))
        time_pck2.append(np.mean(t_pck2_list))
        std_pck2.append(np.std(t_pck2_list))
        time_crt.append(np.mean(t_crt_list))
        std_crt.append(np.std(t_crt_list))

    # Plot
    plt.figure()
    plt.errorbar(disks_list, time_pck0, yerr=std_pck0, fmt='-o', label='PCK0 (CC) + virtual work')
    plt.errorbar(disks_list, time_pck2, yerr=std_pck2, fmt='-s', label='PCK2 + virtual work')
    plt.errorbar(disks_list, time_crt, yerr=std_crt, fmt='-^', label='CRT (shooting-based)')
    plt.xlabel("Number of Disks")
    plt.ylabel("Time (ms)")
    plt.title("Comparison of Shape + Force Estimation Methods")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
