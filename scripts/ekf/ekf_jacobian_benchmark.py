# -*- coding: utf-8 -*-
"""
Benchmark numeric vs analytic Jacobian EKF on the same synthetic data.
Uses wxyz quaternions and normalized arclength (matches shape_EKF_5d_v2).
"""

import argparse
import time

import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm, expm_frechet
from scipy.spatial.transform import Rotation as R


def skew(u):
    return np.array([[0, -u[2], u[1]],
                     [u[2], 0, -u[0]],
                     [-u[1], u[0], 0]])


def phi(s):
    return np.array([[1, s, 0, 0, 0],
                     [0, 0, 1, s, 0],
                     [0, 0, 0, 0, 1]])


def twist(k, e3):
    return np.block([[skew(k), e3[:, None]],
                     [np.zeros((1, 3)), 0]])


def magnus_psi(s_i, h, m, e3):
    xi = np.array([0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6])
    c1, c2 = s_i - h + xi * h
    k1, k2 = phi(c1) @ m, phi(c2) @ m
    e1, e2 = twist(k1, e3), twist(k2, e3)
    return (h / 2) * (e1 + e2) + (np.sqrt(3) / 12) * h ** 2 * (e1 @ e2 - e2 @ e1)


def fwd_rotation(m, s, e3, gamma):
    T = np.eye(4)
    h = s / gamma
    for k in range(1, gamma + 1):
        T = T @ expm(magnus_psi(k * h, h, m, e3))
    return T[:3, :3]


def q_xyzw_to_wxyz(q):
    x, y, z, w = q
    return np.array([w, x, y, z])


def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


def q_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / np.dot(q, q)


def theta(q_meas, m, s, e3, gamma):
    q_pred = q_xyzw_to_wxyz(R.from_matrix(fwd_rotation(m, s, e3, gamma)).as_quat())
    q_e = q_mul(q_meas, q_inv(q_pred))
    return 2 * q_e[1:]


def jac_num(q_meas, m, s, e3, gamma, eps=1e-6):
    J = np.zeros((3, m.size))
    for i in range(m.size):
        m_p, m_m = m.copy(), m.copy()
        m_p[i] += eps
        m_m[i] -= eps
        J[:, i] = (theta(q_meas, m_p, s, e3, gamma)
                   - theta(q_meas, m_m, s, e3, gamma)) / (2 * eps)
    return J


def compute_dtheta_dqhat(q_meas):
    q_meas = q_meas / np.linalg.norm(q_meas)
    q0 = q_meas[0]
    qv = q_meas[1:]
    qv_hat = np.array([[0, -qv[2], qv[1]],
                       [qv[2], 0, -qv[0]],
                       [-qv[1], qv[0], 0]])
    out = np.zeros((3, 4))
    out[:, 0] = qv
    out[:, 1:] = -q0 * np.eye(3) - qv_hat
    return 2 * out


def compute_q_R_derivative(R_matrix):
    q = R.from_matrix(R_matrix).as_quat(scalar_first=True)
    q = q / np.linalg.norm(q)
    q0, q1, q2, q3 = q
    J = np.zeros((4, 3, 3))
    for i in range(3):
        J[0, i, i] = 1 / (8 * q0)
    J[1, 2, 1] = (1 / (4 * q0)) - (q1 / (8 * q0 ** 2))
    J[1, 1, 2] = (-1 / (4 * q0)) - (q1 / (8 * q0 ** 2))
    for i in range(3):
        J[1, i, i] += (-q1 / (8 * q0 ** 2))
    J[2, 0, 2] = (1 / (4 * q0)) - (q2 / (8 * q0 ** 2))
    J[2, 2, 0] = (-1 / (4 * q0)) - (q2 / (8 * q0 ** 2))
    for i in range(3):
        J[2, i, i] += (-q2 / (8 * q0 ** 2))
    J[3, 1, 0] = (1 / (4 * q0)) - (q3 / (8 * q0 ** 2))
    J[3, 0, 1] = (-1 / (4 * q0)) - (q3 / (8 * q0 ** 2))
    for i in range(3):
        J[3, i, i] += (-q3 / (8 * q0 ** 2))
    return J


def compute_R_m_derivative(m_coeffs, s, gamma, e3, dexp_mode="approx"):
    Nm = len(m_coeffs)
    Ns = gamma
    h = s / Ns
    Psi, e_Psi = [], []
    T_before = [np.eye(4)]
    T_after = [np.eye(4)] * (Ns + 1)
    xi = np.array([0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6])
    dPsi_dm = [np.zeros((4, 4, Nm)) for _ in range(Ns)]
    de_Psi_dm = [np.zeros((4, 4, Nm)) for _ in range(Ns)]

    for k in range(1, Ns + 1):
        s_i = k * h
        Psi_k = magnus_psi(s_i, h, m_coeffs, e3)
        Psi.append(Psi_k)
        e_Psi_k = expm(Psi_k)
        e_Psi.append(e_Psi_k)
        T_before.append(T_before[-1] @ e_Psi_k)

    T_after[Ns] = np.eye(4)
    for k in range(Ns - 1, 0, -1):
        T_after[k] = e_Psi[k] @ T_after[k + 1]

    for k in range(1, Ns + 1):
        s_i = k * h
        c1 = s_i - h + xi[0] * h
        c2 = s_i - h + xi[1] * h
        phi_c1 = phi(c1)
        phi_c2 = phi(c2)
        kappa_1 = phi_c1 @ m_coeffs
        kappa_2 = phi_c2 @ m_coeffs
        eta_1 = twist(kappa_1, e3)
        eta_2 = twist(kappa_2, e3)
        for i in range(Nm):
            dkappa_1 = phi_c1[:, i]
            dkappa_2 = phi_c2[:, i]
            deta_1 = twist(dkappa_1, np.zeros(3))
            deta_2 = twist(dkappa_2, np.zeros(3))
            comm_deta = (deta_1 @ eta_2 - eta_2 @ deta_1) + (eta_1 @ deta_2 - deta_2 @ eta_1)
            dPsi_k = (h / 2) * (deta_1 + deta_2) + (np.sqrt(3) / 12) * h ** 2 * comm_deta
            dPsi_dm[k - 1][:, :, i] = dPsi_k

    for k in range(Ns):
        Psi_k = Psi[k]
        e_Psi_k = e_Psi[k]
        for i in range(Nm):
            dPsi_k = dPsi_dm[k][:, :, i]
            if dexp_mode == "exact":
                # Exact differential via Frechet derivative of expm.
                _, dE = expm_frechet(Psi_k, dPsi_k, compute_expm=True)
                de_Psi_dm[k][:, :, i] = dE
            else:
                # 1st-order dexp approximation: dexp(A)·dA ≈ dA + 0.5[A, dA]
                # This is cheaper but less accurate than exact SE(3) dexp.
                dexpinv = dPsi_k + 0.5 * (Psi_k @ dPsi_k - dPsi_k @ Psi_k)
                de_Psi_dm[k][:, :, i] = e_Psi_k @ dexpinv

    dT_dm = np.zeros((4, 4, Nm))
    for i in range(Nm):
        for k in range(Ns):
            dT_dm[:, :, i] += T_before[k] @ de_Psi_dm[k][:, :, i] @ T_after[k + 1]
    return dT_dm[:3, :3, :]


def jac_analy(q_meas, m, s, e3, gamma, dexp_mode="approx"):
    dtheta_dqhat = compute_dtheta_dqhat(q_meas)
    R_matrix = fwd_rotation(m, s, e3, gamma)
    dqhat_dR = compute_q_R_derivative(R_matrix)
    dR_dm = compute_R_m_derivative(m, s, gamma, e3, dexp_mode=dexp_mode)
    dtheta_dR = np.einsum("ik,kjl->ijl", dtheta_dqhat, dqhat_dR)
    return np.einsum("ijl,jln->in", dtheta_dR, dR_dm)


def run_ekf(meas_seq, m_true, imu_pos, e3, gamma, use_analytic, dexp_mode="approx"):
    state_dim = m_true.size
    proc_std0, proc_std1 = 1e-5, 1e-6
    Q = np.diag([proc_std0**2, proc_std1**2,
                 proc_std0**2, proc_std1**2,
                 proc_std0**2])
    meas_std_deg = 0.5
    R_single = (np.deg2rad(meas_std_deg) ** 2) * np.eye(3)

    m_est = np.zeros(state_dim)
    P_est = 1e-2 * np.eye(state_dim)
    hist = []

    t0 = time.perf_counter()
    for k in range(len(meas_seq)):
        m_pred = m_est.copy()
        P_pred = P_est + Q
        H_stack, r_stack = [], []
        for i, s in enumerate(imu_pos):
            q_meas = meas_seq[k][i]
            r = -theta(q_meas, m_pred, s, e3, gamma)
            if use_analytic:
                H = jac_analy(q_meas, m_pred, s, e3, gamma, dexp_mode=dexp_mode)
            else:
                H = jac_num(q_meas, m_pred, s, e3, gamma)
            H_stack.append(H)
            r_stack.append(r)
        H = np.vstack(H_stack)
        r = np.hstack(r_stack)
        R_big = np.kron(np.eye(len(imu_pos)), R_single)
        S = H @ P_pred @ H.T + R_big
        K = P_pred @ H.T @ inv(S)
        m_est = m_pred + K @ r
        P_est = (np.eye(state_dim) - K @ H) @ P_pred
        hist.append(m_est.copy())
    t1 = time.perf_counter()

    hist = np.array(hist)
    err = hist - m_true
    rmse_t = np.sqrt(np.mean(err ** 2, axis=1))
    out = {
        "m_est": m_est,
        "hist": hist,
        "rmse_final": float(np.sqrt(np.mean((m_est - m_true) ** 2))),
        "rmse_mean": float(np.mean(rmse_t)),
        "rmse_max": float(np.max(rmse_t)),
        "mae_mean": float(np.mean(np.abs(err))),
        "time_total": float(t1 - t0),
        "time_per_step": float((t1 - t0) / len(meas_seq)),
    }
    return out


def build_meas_seq(m_true, imu_pos, e3, gamma, steps, meas_std_deg, rng):
    meas_seq = []
    for _ in range(steps):
        frame = []
        for s in imu_pos:
            q_clean = q_xyzw_to_wxyz(
                R.from_matrix(fwd_rotation(m_true, s, e3, gamma)).as_quat()
            )
            noise = R.from_rotvec(rng.randn(3) * np.deg2rad(meas_std_deg))
            q_meas = q_mul(q_xyzw_to_wxyz(noise.as_quat()), q_clean)
            if q_meas[0] < 0:
                q_meas = -q_meas
            frame.append(q_meas)
        meas_seq.append(frame)
    return meas_seq


def summarize(results):
    keys = ["rmse_final", "rmse_mean", "rmse_max", "mae_mean", "time_total", "time_per_step"]
    stats = {}
    for k in keys:
        vals = np.array([r[k] for r in results])
        stats[k] = (float(np.mean(vals)), float(np.std(vals)))
    return stats


def print_table(stats_num, stats_ana, stats_ana_exact=None):
    rows = [
        ("RMSE final", "rmse_final"),
        ("RMSE mean", "rmse_mean"),
        ("RMSE max", "rmse_max"),
        ("MAE mean", "mae_mean"),
        ("Time total (s)", "time_total"),
        ("Time/step (s)", "time_per_step"),
    ]
    print("\nJacobian Benchmark (mean ± std)")
    print("-" * 72)
    if stats_ana_exact is None:
        print(f"{'Metric':<20} | {'Numeric':<22} | {'Analytic':<22}")
    else:
        print(f"{'Metric':<20} | {'Numeric':<22} | {'Analytic (approx)':<22} | {'Analytic (exact)':<22}")
    print("-" * 72)
    for name, key in rows:
        n_mu, n_sd = stats_num[key]
        a_mu, a_sd = stats_ana[key]
        if stats_ana_exact is None:
            print(f"{name:<20} | {n_mu:>9.4e} ± {n_sd:<9.2e} | {a_mu:>9.4e} ± {a_sd:<9.2e}")
        else:
            ax_mu, ax_sd = stats_ana_exact[key]
            print(
                f"{name:<20} | {n_mu:>9.4e} ± {n_sd:<9.2e} | "
                f"{a_mu:>9.4e} ± {a_sd:<9.2e} | "
                f"{ax_mu:>9.4e} ± {ax_sd:<9.2e}"
            )
    print("-" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--gamma", type=int, default=20)
    parser.add_argument("--seed-list", type=str, default="11,22,33,44,55,66,77,88,99,111")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--dexp", choices=["approx", "exact"], default="approx",
                        help="Analytic Jacobian dexp mode (default: approx).")
    parser.add_argument("--compare-dexp", action="store_true",
                        help="Run analytic Jacobian with both approx and exact dexp.")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seed_list.split(",") if s.strip()]
    if len(seeds) < args.trials:
        raise ValueError("seed-list must have at least as many entries as trials")

    imu_pos = np.array([0.25, 0.50, 0.75])
    L_phys = 100.0
    e3 = np.array([0, 0, L_phys])
    meas_std_deg = 0.5

    results_num = []
    results_ana = []
    results_ana_exact = []
    rmse_t_num = []
    rmse_t_ana = []

    for t in range(args.trials):
        rng = np.random.RandomState(seeds[t])
        m_true = rng.uniform(-2, 2, 5)
        meas_seq = build_meas_seq(
            m_true, imu_pos, e3, args.gamma, args.steps, meas_std_deg, rng
        )
        out_num = run_ekf(meas_seq, m_true, imu_pos, e3, args.gamma, False)
        out_ana = run_ekf(
            meas_seq, m_true, imu_pos, e3, args.gamma, True, dexp_mode=args.dexp
        )
        results_num.append(out_num)
        results_ana.append(out_ana)
        if args.compare_dexp:
            out_ana_exact = run_ekf(
                meas_seq, m_true, imu_pos, e3, args.gamma, True, dexp_mode="exact"
            )
            results_ana_exact.append(out_ana_exact)

        rmse_t_num.append(np.sqrt(np.mean((out_num["hist"] - m_true) ** 2, axis=1)))
        rmse_t_ana.append(np.sqrt(np.mean((out_ana["hist"] - m_true) ** 2, axis=1)))

    stats_num = summarize(results_num)
    stats_ana = summarize(results_ana)
    if args.compare_dexp:
        stats_ana_exact = summarize(results_ana_exact)
        print_table(stats_num, stats_ana, stats_ana_exact)
    else:
        print_table(stats_num, stats_ana)

    if args.plot:
        rmse_t_num = np.mean(np.vstack(rmse_t_num), axis=0)
        rmse_t_ana = np.mean(np.vstack(rmse_t_ana), axis=0)
        import matplotlib.pyplot as plt
        t = np.arange(args.steps)
        plt.plot(t, rmse_t_num, label="numeric")
        plt.plot(t, rmse_t_ana, label="analytic")
        plt.xlabel("time step")
        plt.ylabel("RMSE(m)")
        plt.title("EKF RMSE over time (mean across trials)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
