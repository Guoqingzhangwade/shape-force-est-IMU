# -*- coding: utf-8 -*-
"""
Compare EKF measurement models (synthetic benchmark):
  1) Quaternion residual with numeric Jacobian
  2) Quaternion residual with analytic Jacobian (known to be sensitive to R->q branch logic)
  3) SO(3) log residual with analytic Jacobian (Magnus + 1st-order dexp approx)

Key upgrades vs your previous script:
  - SO(3) uses a numerically-stable Jr^{-1} coefficient via cot(theta/2) (robust near theta ~ pi).
  - SO(3) supports tuning via a scalar measurement covariance scale: R_so3 = scale * sigma^2 I.
  - Optional NIS reporting (consistency check) for each model.
  - Enforces a consistent quaternion sign convention (w >= 0) for q_pred and dq/dR (helps avoid sign flips).

Usage examples:
  python compare_models_v2.py --compare-so3-scales --so3-scale-list "0.5,1,2,3"
  python compare_models_v2.py --trials 10 --steps 100 --gamma 20 --plot
"""

import argparse
import time
import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R


# ----------------------------- Lie helpers -----------------------------

def skew(u):
    return np.array([[0.0, -u[2],  u[1]],
                     [u[2],  0.0, -u[0]],
                     [-u[1], u[0], 0.0]], dtype=float)


def vee(S):
    # vee of a skew-symmetric matrix
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def so3_log(Rm):
    # rotation matrix -> rotvec (principal, angle in [0, pi])
    return R.from_matrix(Rm).as_rotvec()


def jr_inv_so3(r):
    """
    Right Jacobian inverse on SO(3): Jr^{-1}(r)
    Stable coefficient using:
      (1+cos θ)/(2 θ sin θ) = 0.5 * cot(θ/2) / θ
      a = 1/θ^2 - (1+cos θ)/(2 θ sin θ) = (1 - 0.5 θ cot(θ/2))/θ^2

    Uses series near θ -> 0.
    """
    theta = float(np.linalg.norm(r))
    A = skew(r)

    if theta < 1e-8:
        # series: I + 1/2 A + 1/12 A^2
        return np.eye(3) + 0.5 * A + (1.0 / 12.0) * (A @ A)

    half = 0.5 * theta
    sin_half = np.sin(half)
    cos_half = np.cos(half)

    # In normal use, sin_half is not tiny when theta ~ pi (sin(pi/2)=1),
    # but guard anyway.
    if abs(sin_half) < 1e-12:
        # fallback to series-ish clamp
        return np.eye(3) + 0.5 * A + (1.0 / 12.0) * (A @ A)

    cot_half = cos_half / sin_half
    a = (1.0 - 0.5 * theta * cot_half) / (theta ** 2)
    return np.eye(3) + 0.5 * A + a * (A @ A)


# ------------------------ Kinematics + derivatives ------------------------

def phi(s):
    # 3x5 for PCK2-style in your benchmark
    return np.array([[1, s, 0, 0, 0],
                     [0, 0, 1, s, 0],
                     [0, 0, 0, 0, 1]], dtype=float)


def twist(k, e3):
    return np.block([[skew(k), e3[:, None]],
                     [np.zeros((1, 3)), 0.0]])


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


def compute_R_m_derivative(m_coeffs, s, gamma, e3):
    """
    Returns dR/dm as a tensor 3x3xNm using:
      - Magnus for Psi_k
      - 1st-order dexp approximation for d exp(Psi_k)
    """
    Nm = len(m_coeffs)
    Ns = gamma
    h = s / Ns

    Psi, e_Psi = [], []
    T_before = [np.eye(4)]
    T_after = [np.eye(4)] * (Ns + 1)

    xi = np.array([0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6])
    dPsi_dm = [np.zeros((4, 4, Nm)) for _ in range(Ns)]
    de_Psi_dm = [np.zeros((4, 4, Nm)) for _ in range(Ns)]

    # forward cache
    for k in range(1, Ns + 1):
        s_i = k * h
        Psi_k = magnus_psi(s_i, h, m_coeffs, e3)
        Psi.append(Psi_k)
        e_Psi_k = expm(Psi_k)
        e_Psi.append(e_Psi_k)
        T_before.append(T_before[-1] @ e_Psi_k)

    # backward cache
    T_after[Ns] = np.eye(4)
    for k in range(Ns - 1, 0, -1):
        T_after[k] = e_Psi[k] @ T_after[k + 1]

    # dPsi/dm
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

    # d exp(Psi)/dm using 1st-order dexp approx
    for k in range(Ns):
        Psi_k = Psi[k]
        e_Psi_k = e_Psi[k]
        for i in range(Nm):
            dPsi_k = dPsi_dm[k][:, :, i]
            dexpinv = dPsi_k + 0.5 * (Psi_k @ dPsi_k - dPsi_k @ Psi_k)
            de_Psi_dm[k][:, :, i] = e_Psi_k @ dexpinv

    # accumulate dT/dm
    dT_dm = np.zeros((4, 4, Nm))
    for i in range(Nm):
        for k in range(Ns):
            dT_dm[:, :, i] += T_before[k] @ de_Psi_dm[k][:, :, i] @ T_after[k + 1]

    return dT_dm[:3, :3, :]


# ----------------------------- Quaternion residual -----------------------------

def q_xyzw_to_wxyz(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array([w, x, y, z], dtype=float)


def q_wxyz_to_xyzw(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([x, y, z, w], dtype=float)


def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=float)


def q_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float) / np.dot(q, q)


def theta_quat(q_meas_wxyz, m, s, e3, gamma):
    R_pred = fwd_rotation(m, s, e3, gamma)
    q_pred = q_xyzw_to_wxyz(R.from_matrix(R_pred).as_quat())  # xyzw -> wxyz

    # enforce consistent sign (w>=0) to reduce discontinuities
    if q_pred[0] < 0:
        q_pred = -q_pred

    q_e = q_mul(q_meas_wxyz, q_inv(q_pred))
    return 2.0 * q_e[1:]


def jac_num_quat(q_meas_wxyz, m, s, e3, gamma, eps=1e-6):
    J = np.zeros((3, m.size))
    for i in range(m.size):
        m_p, m_m = m.copy(), m.copy()
        m_p[i] += eps
        m_m[i] -= eps
        J[:, i] = (theta_quat(q_meas_wxyz, m_p, s, e3, gamma)
                   - theta_quat(q_meas_wxyz, m_m, s, e3, gamma)) / (2 * eps)
    return J


def compute_dtheta_dqhat(q_meas_wxyz):
    q_meas = q_meas_wxyz / np.linalg.norm(q_meas_wxyz)
    q0 = q_meas[0]
    qv = q_meas[1:]
    qv_hat = skew(qv)

    out = np.zeros((3, 4))
    out[:, 0] = qv
    out[:, 1:] = -q0 * np.eye(3) - qv_hat
    return 2.0 * out


def compute_q_R_derivative_trace_branch(R_matrix):
    """
    WARNING: This matches the trace-based closed form, not SciPy's full branch logic.
    We keep it only for reference. Sign-fix is applied consistently.

    Returns tensor dq/dR of shape (4,3,3) for q in wxyz.
    """
    q = R.from_matrix(R_matrix).as_quat(scalar_first=True)  # wxyz
    q = q / np.linalg.norm(q)

    # enforce consistent sign; derivative flips with q
    sign = 1.0
    if q[0] < 0:
        q = -q
        sign = -1.0

    q0, q1, q2, q3 = q
    # avoid blow-up if q0 ~ 0 in this trace-branch formula
    if abs(q0) < 1e-10:
        # return something finite; EKF will likely behave poorly anyway in this branch
        return np.zeros((4, 3, 3))

    J = np.zeros((4, 3, 3))
    for i in range(3):
        J[0, i, i] = 1.0 / (8.0 * q0)

    J[1, 2, 1] = (1.0 / (4.0 * q0)) - (q1 / (8.0 * q0 ** 2))
    J[1, 1, 2] = (-1.0 / (4.0 * q0)) - (q1 / (8.0 * q0 ** 2))
    for i in range(3):
        J[1, i, i] += (-q1 / (8.0 * q0 ** 2))

    J[2, 0, 2] = (1.0 / (4.0 * q0)) - (q2 / (8.0 * q0 ** 2))
    J[2, 2, 0] = (-1.0 / (4.0 * q0)) - (q2 / (8.0 * q0 ** 2))
    for i in range(3):
        J[2, i, i] += (-q2 / (8.0 * q0 ** 2))

    J[3, 1, 0] = (1.0 / (4.0 * q0)) - (q3 / (8.0 * q0 ** 2))
    J[3, 0, 1] = (-1.0 / (4.0 * q0)) - (q3 / (8.0 * q0 ** 2))
    for i in range(3):
        J[3, i, i] += (-q3 / (8.0 * q0 ** 2))

    # derivative flips if q flips
    return sign * J


def jac_analy_quat_reference(q_meas_wxyz, m, s, e3, gamma):
    dtheta_dqhat = compute_dtheta_dqhat(q_meas_wxyz)
    R_pred = fwd_rotation(m, s, e3, gamma)
    dqhat_dR = compute_q_R_derivative_trace_branch(R_pred)
    dR_dm = compute_R_m_derivative(m, s, gamma, e3)

    dtheta_dR = np.einsum("ik,kjl->ijl", dtheta_dqhat, dqhat_dR)  # 3x3x3
    return np.einsum("ijl,jln->in", dtheta_dR, dR_dm)            # 3xNm


# ----------------------------- SO(3) residual -----------------------------

def measurement_so3_and_H(m, s_vals, R_meas_list, e3, gamma):
    """
    Returns:
      H: (3*Nsensors, Nm)
      r: (3*Nsensors,)
    """
    H_blocks, r_blocks = [], []

    for s_i, R_meas in zip(s_vals, R_meas_list):
        R_pred = fwd_rotation(m, s_i, e3, gamma)
        R_e = R_meas @ R_pred.T
        r = so3_log(R_e)
        Jr_inv = jr_inv_so3(r)

        dR_dm = compute_R_m_derivative(m, s_i, gamma, e3)

        H_i = np.zeros((3, m.size))
        for p in range(m.size):
            dR = dR_dm[:, :, p]
            # left-trivialized perturbation: dphi = vee(dR * R^T)  (skew part)
            A = dR @ R_pred.T
            A = 0.5 * (A - A.T)
            H_i[:, p] = -Jr_inv @ vee(A)

        H_blocks.append(H_i)
        r_blocks.append(r)

    return np.vstack(H_blocks), np.hstack(r_blocks)


# ----------------------------- EKF runners -----------------------------

def ekf_quat(meas_q_seq, m_true, imu_pos, e3, gamma, mode, meas_std_deg):
    """
    mode:
      - "quat_numeric"
      - "quat_analytic" (reference only; not branch-matched to SciPy)
    """
    state_dim = m_true.size

    proc_std0, proc_std1 = 1e-5, 1e-6
    Q = np.diag([proc_std0**2, proc_std1**2,
                 proc_std0**2, proc_std1**2,
                 proc_std0**2])

    sigma = np.deg2rad(meas_std_deg)
    R_single = (sigma ** 2) * np.eye(3)

    m_est = np.zeros(state_dim)
    P_est = 1e-2 * np.eye(state_dim)
    hist = []
    nis_hist = []

    t0 = time.perf_counter()
    for k in range(len(meas_q_seq)):
        m_pred = m_est.copy()
        P_pred = P_est + Q

        H_stack, innov_stack = [], []
        for i, s in enumerate(imu_pos):
            q_meas = meas_q_seq[k][i]
            r = theta_quat(q_meas, m_pred, s, e3, gamma)  # h(x)
            innov = -r                                    # z-h with z=0

            if mode == "quat_numeric":
                H = jac_num_quat(q_meas, m_pred, s, e3, gamma)
            elif mode == "quat_analytic":
                H = jac_analy_quat_reference(q_meas, m_pred, s, e3, gamma)
            else:
                raise ValueError(mode)

            H_stack.append(H)
            innov_stack.append(innov)

        H = np.vstack(H_stack)
        innov = np.hstack(innov_stack)

        R_big = np.kron(np.eye(len(imu_pos)), R_single)
        S = H @ P_pred @ H.T + R_big
        K = P_pred @ H.T @ inv(S)

        m_est = m_pred + K @ innov
        P_est = (np.eye(state_dim) - K @ H) @ P_pred
        hist.append(m_est.copy())

        nis_hist.append(float(innov.T @ inv(S) @ innov))

    t1 = time.perf_counter()

    hist = np.array(hist)
    err = hist - m_true
    rmse_t = np.sqrt(np.mean(err ** 2, axis=1))

    return {
        "m_est": m_est,
        "hist": hist,
        "rmse_final": float(np.sqrt(np.mean((m_est - m_true) ** 2))),
        "rmse_mean": float(np.mean(rmse_t)),
        "rmse_max": float(np.max(rmse_t)),
        "mae_mean": float(np.mean(np.abs(err))),
        "time_total": float(t1 - t0),
        "time_per_step": float((t1 - t0) / len(meas_q_seq)),
        "nis_mean": float(np.mean(nis_hist)),
        "nis_std": float(np.std(nis_hist)),
    }


def ekf_so3(meas_R_seq, m_true, imu_pos, e3, gamma, meas_std_deg, R_scale_so3=1.0):
    state_dim = m_true.size

    proc_std0, proc_std1 = 1e-5, 1e-6
    Q = np.diag([proc_std0**2, proc_std1**2,
                 proc_std0**2, proc_std1**2,
                 proc_std0**2])

    sigma = np.deg2rad(meas_std_deg)
    R_single = R_scale_so3 * (sigma ** 2) * np.eye(3)

    m_est = np.zeros(state_dim)
    P_est = 1e-2 * np.eye(state_dim)
    hist = []
    nis_hist = []

    t0 = time.perf_counter()
    for k in range(len(meas_R_seq)):
        m_pred = m_est.copy()
        P_pred = P_est + Q

        H, r = measurement_so3_and_H(m_pred, imu_pos, meas_R_seq[k], e3, gamma)
        innov = -r  # z-h, z=0

        R_big = np.kron(np.eye(len(imu_pos)), R_single)
        S = H @ P_pred @ H.T + R_big
        K = P_pred @ H.T @ inv(S)

        m_est = m_pred + K @ innov
        P_est = (np.eye(state_dim) - K @ H) @ P_pred
        hist.append(m_est.copy())

        nis_hist.append(float(innov.T @ inv(S) @ innov))

    t1 = time.perf_counter()

    hist = np.array(hist)
    err = hist - m_true
    rmse_t = np.sqrt(np.mean(err ** 2, axis=1))

    return {
        "m_est": m_est,
        "hist": hist,
        "rmse_final": float(np.sqrt(np.mean((m_est - m_true) ** 2))),
        "rmse_mean": float(np.mean(rmse_t)),
        "rmse_max": float(np.max(rmse_t)),
        "mae_mean": float(np.mean(np.abs(err))),
        "time_total": float(t1 - t0),
        "time_per_step": float((t1 - t0) / len(meas_R_seq)),
        "nis_mean": float(np.mean(nis_hist)),
        "nis_std": float(np.std(nis_hist)),
    }


# ----------------------------- Data generation -----------------------------

def build_meas_seq(m_true, imu_pos, e3, gamma, steps, meas_std_deg, rng):
    """
    Generates measurement sequences:
      meas_q[t][i] : wxyz quaternion with w>=0
      meas_R[t][i] : corresponding rotation matrix
    """
    meas_q = []
    meas_R = []

    for _ in range(steps):
        frame_q = []
        frame_R = []

        for s in imu_pos:
            R_clean = fwd_rotation(m_true, s, e3, gamma)
            q_clean = q_xyzw_to_wxyz(R.from_matrix(R_clean).as_quat())  # wxyz

            # multiplicative noise in SO(3) via rotvec
            noise = R.from_rotvec(rng.randn(3) * np.deg2rad(meas_std_deg))
            q_noise = q_xyzw_to_wxyz(noise.as_quat())  # wxyz

            q_meas = q_mul(q_noise, q_clean)
            if q_meas[0] < 0:
                q_meas = -q_meas

            frame_q.append(q_meas)
            frame_R.append(R.from_quat(q_wxyz_to_xyzw(q_meas)).as_matrix())

        meas_q.append(frame_q)
        meas_R.append(frame_R)

    return meas_q, meas_R


# ----------------------------- Reporting -----------------------------

def summarize(results):
    keys = ["rmse_final", "rmse_mean", "rmse_max", "mae_mean",
            "time_total", "time_per_step", "nis_mean", "nis_std"]
    stats = {}
    for k in keys:
        vals = np.array([r[k] for r in results], dtype=float)
        stats[k] = (float(np.mean(vals)), float(np.std(vals)))
    return stats


def print_table(stats_qnum, stats_qana, stats_so3, so3_scale, nis_dof):
    rows = [
        ("RMSE final", "rmse_final"),
        ("RMSE mean", "rmse_mean"),
        ("RMSE max", "rmse_max"),
        ("MAE mean", "mae_mean"),
        ("Time total (s)", "time_total"),
        ("Time/step (s)", "time_per_step"),
        ("NIS mean", "nis_mean"),
        ("NIS std", "nis_std"),
    ]

    print("\nMeasurement Model Comparison (mean ± std)")
    print("-" * 104)
    print(f"{'Metric':<20} | {'Quat numeric':<24} | {'Quat analytic(ref)':<24} | {'SO(3) analytic':<24}")
    print("-" * 104)

    for name, key in rows:
        n_mu, n_sd = stats_qnum[key]
        a_mu, a_sd = stats_qana[key]
        s_mu, s_sd = stats_so3[key]
        print(
            f"{name:<20} | {n_mu:>10.4e} ± {n_sd:<10.2e} | "
            f"{a_mu:>10.4e} ± {a_sd:<10.2e} | "
            f"{s_mu:>10.4e} ± {s_sd:<10.2e}"
        )

    print("-" * 104)
    print(f"SO(3) R-scale used: {so3_scale}")
    print(f"NIS DOF (per step): {nis_dof}  (target: NIS mean ≈ DOF if tuning is consistent)")
    print("-" * 104)


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--gamma", type=int, default=20)
    parser.add_argument("--seed-list", type=str, default="11,22,33,44,55,66,77,88,99,111")
    parser.add_argument("--meas-std-deg", type=float, default=0.5)

    parser.add_argument("--so3-scale", type=float, default=1.0,
                        help="SO(3) measurement covariance scale: R_so3 = scale * sigma^2 I")
    parser.add_argument("--compare-so3-scales", action="store_true",
                        help="Sweep --so3-scale-list and pick best by --objective")
    parser.add_argument("--so3-scale-list", type=str, default="0.5,1.0,1.5,2.0,3.0")
    parser.add_argument("--objective", choices=["rmse_mean", "rmse_final"], default="rmse_mean")

    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seed_list.split(",") if s.strip()]
    if len(seeds) < args.trials:
        raise ValueError("seed-list must have at least as many entries as trials")

    imu_pos = np.array([0.25, 0.50, 0.75], dtype=float)
    L_phys = 100.0
    e3 = np.array([0.0, 0.0, L_phys], dtype=float)

    # dof for NIS: 3 per IMU
    nis_dof = 3 * len(imu_pos)

    # helper to run one full benchmark for a given SO3 scale
    def run_for_scale(so3_scale):
        results_qnum, results_qana, results_so3 = [], [], []
        rmse_t_qnum, rmse_t_qana, rmse_t_so3 = [], [], []

        for t in range(args.trials):
            rng = np.random.RandomState(seeds[t])
            m_true = rng.uniform(-2, 2, 5)

            meas_q, meas_R = build_meas_seq(
                m_true, imu_pos, e3, args.gamma, args.steps, args.meas_std_deg, rng
            )

            out_qnum = ekf_quat(meas_q, m_true, imu_pos, e3, args.gamma, "quat_numeric", args.meas_std_deg)
            out_qana = ekf_quat(meas_q, m_true, imu_pos, e3, args.gamma, "quat_analytic", args.meas_std_deg)
            out_so3 = ekf_so3(meas_R, m_true, imu_pos, e3, args.gamma, args.meas_std_deg, R_scale_so3=so3_scale)

            results_qnum.append(out_qnum)
            results_qana.append(out_qana)
            results_so3.append(out_so3)

            rmse_t_qnum.append(np.sqrt(np.mean((out_qnum["hist"] - m_true) ** 2, axis=1)))
            rmse_t_qana.append(np.sqrt(np.mean((out_qana["hist"] - m_true) ** 2, axis=1)))
            rmse_t_so3.append(np.sqrt(np.mean((out_so3["hist"] - m_true) ** 2, axis=1)))

        return {
            "stats_qnum": summarize(results_qnum),
            "stats_qana": summarize(results_qana),
            "stats_so3": summarize(results_so3),
            "rmse_t_qnum": np.mean(np.vstack(rmse_t_qnum), axis=0),
            "rmse_t_qana": np.mean(np.vstack(rmse_t_qana), axis=0),
            "rmse_t_so3": np.mean(np.vstack(rmse_t_so3), axis=0),
        }

    if args.compare_so3_scales:
        scale_list = [float(x) for x in args.so3_scale_list.split(",") if x.strip()]
        sweep = []
        for sc in scale_list:
            out = run_for_scale(sc)
            obj = out["stats_so3"][args.objective][0]  # mean
            sweep.append((obj, sc, out))

        sweep.sort(key=lambda x: x[0])
        best_obj, best_scale, best_out = sweep[0]

        print("\nSO(3) scale sweep results (lower is better):")
        for obj, sc, _ in sweep:
            print(f"  scale={sc:<6g}  {args.objective}={obj:.6e}")
        print(f"\nBest scale={best_scale} with {args.objective}={best_obj:.6e}")

        stats_qnum = best_out["stats_qnum"]
        stats_qana = best_out["stats_qana"]
        stats_so3 = best_out["stats_so3"]
        print_table(stats_qnum, stats_qana, stats_so3, best_scale, nis_dof)

        if args.plot:
            import matplotlib.pyplot as plt
            t = np.arange(args.steps)
            plt.plot(t, best_out["rmse_t_qnum"], label="quat numeric")
            plt.plot(t, best_out["rmse_t_qana"], label="quat analytic(ref)")
            plt.plot(t, best_out["rmse_t_so3"], label=f"so3 analytic (scale={best_scale:g})")
            plt.xlabel("time step")
            plt.ylabel("RMSE(m)")
            plt.title("EKF RMSE over time (mean across trials)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        return

    # single run at provided so3 scale
    out = run_for_scale(args.so3_scale)
    print_table(out["stats_qnum"], out["stats_qana"], out["stats_so3"], args.so3_scale, nis_dof)

    if args.plot:
        import matplotlib.pyplot as plt
        t = np.arange(args.steps)
        plt.plot(t, out["rmse_t_qnum"], label="quat numeric")
        plt.plot(t, out["rmse_t_qana"], label="quat analytic(ref)")
        plt.plot(t, out["rmse_t_so3"], label=f"so3 analytic (scale={args.so3_scale:g})")
        plt.xlabel("time step")
        plt.ylabel("RMSE(m)")
        plt.title("EKF RMSE over time (mean across trials)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

