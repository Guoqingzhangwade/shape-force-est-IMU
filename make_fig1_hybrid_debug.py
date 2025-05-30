#!/usr/bin/env python3
"""
debug_one_sample.py  –  run an IEKF on a single TDCR sample
                        and print / plot error vs. iteration.

Usage
-----
python debug_one_sample.py --idx 0 \
                           --gt  tdcr_gt_samples.npz \
                           --meas tdcr_meas_samples.npz \
                           --max-iter 30
"""
from __future__ import annotations
import argparse, json
import numpy as np
from numpy.linalg import inv, norm
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm
import matplotlib.pyplot as plt
import ipdb

# ---------------- EKF model parameters (same as before) --------------------
STATE_DIM    = 6
IMU_S        = np.array([12.0/39, 25.0/39, 1.00])
L_PHYS_MM    = 100.0
GAMMA        = 20
PROC_STD0, PROC_STD1 = 1e-5, 1e-6
Q = np.diag([PROC_STD0**2, PROC_STD1**2,
             PROC_STD0**2, PROC_STD1**2,
             PROC_STD0**2, PROC_STD1**2])
MEAS_STD_DEG = 1.0
R_SINGLE     = (np.deg2rad(MEAS_STD_DEG)**2) * np.eye(3)
INIT_M       = np.ones(STATE_DIM)

# ---------------- helpers --------------------------------------------------
def skew(u):
    return np.array([[0, -u[2], u[1]],
                     [u[2],  0, -u[0]],
                     [-u[1], u[0], 0]])

def phi(s):
    return np.array([[1, s, 0, 0, 0, 0],
                     [0, 0, 1, s, 0, 0],
                     [0, 0, 0, 0, 1, s]])

def twist(k, e3):
    return np.block([[skew(k), e3[:, None]],
                     [np.zeros((1, 3)), 0]])

def magnus_psi(s_i, h, m, e3):
    xi = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
    c1, c2 = s_i - h + xi*h
    k1, k2 = phi(c1)@m, phi(c2)@m
    e1, e2 = twist(k1,e3), twist(k2,e3)
    return (h/2)*(e1+e2) + (np.sqrt(3)/12)*h**2*(e1@e2 - e2@e1)

def fwd_frame(m, s, e3, gamma=GAMMA):
    """Return 4×4 frame at arclength s.  Translation kept in *metres*."""
    T = np.eye(4); h = s/gamma
    for k in range(1, gamma+1):
        T = T @ expm(magnus_psi(k*h, h, m, e3))
    return T

def q_xyzw_to_wxyz(q):
    x,y,z,w = q
    return np.array([w,x,y,z])

def q_mul(a,b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def q_inv(q):
    w,x,y,z = q
    return np.array([w,-x,-y,-z]) / np.dot(q,q)

def theta(q_meas, m, s, e3):
    q_pred = q_xyzw_to_wxyz(R.from_matrix(fwd_frame(m,s,e3)[:3,:3]).as_quat())
    if q_pred[0] < 0: q_pred = -q_pred
    q_e = q_mul(q_meas, q_inv(q_pred))
    if q_e[0] < 0: q_e = -q_e
    return 2*q_e[1:]

def jac_num(q_meas, m, s, e3, eps=1e-6):
    J = np.zeros((3, STATE_DIM))
    for i in range(STATE_DIM):
        m_p, m_m = m.copy(), m.copy()
        m_p[i] += eps; m_m[i] -= eps
        J[:, i] = (theta(q_meas, m_p, s, e3) -
                   theta(q_meas, m_m, s, e3)) / (2*eps)
    return J
# --------------------------------------------------------------------------

def run_iterated_EKF(q_meas_frame, max_iter=30, tol=1e-6):
    e3 = np.array([0,0,L_PHYS_MM])   # translation scaling vector (mm)

    m_est = INIT_M.copy()
    P_est = 1e-1*np.eye(STATE_DIM)

    err_log = []                     # innovation norm per iteration

    R_big = np.kron(np.eye(len(IMU_S)), R_SINGLE)
    for it in range(max_iter):
        # prediction (static)
        m_pred, P_pred = m_est, P_est + Q

        H_list, r_list = [], []
        for s, q_meas in zip(IMU_S, q_meas_frame):
            r = -theta(q_meas, m_pred, s, e3)
            H = jac_num(q_meas, m_pred, s, e3)
            H_list.append(H); r_list.append(r)
        H = np.vstack(H_list);  r = np.hstack(r_list)

        S = H @ P_pred @ H.T + R_big
        K = P_pred @ H.T @ np.linalg.inv(S)

        m_new = m_pred + K @ r
        P_new = (np.eye(STATE_DIM) - K @ H) @ P_pred

        err_log.append(norm(r))

        if norm(m_new - m_est) < tol:
            m_est, P_est = m_new, P_new
            break
        m_est, P_est = m_new, P_new

    return m_est, err_log

# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=6, help="sample index")
    parser.add_argument("--gt",  default="tdcr_gt_samples.npz")
    parser.add_argument("--meas",default="tdcr_meas_samples.npz")
    parser.add_argument("--max-iter", type=int, default=30)
    args = parser.parse_args()

    gt   = np.load(args.gt,   allow_pickle=True)
    meas = np.load(args.meas, allow_pickle=True)

    T_gt   = gt["T"][args.idx]            # (n_disks,4,4)
    q_meas = meas["q_meas"][args.idx]     # (3,4)

    m_hat, innov_log = run_iterated_EKF(q_meas,
                                        max_iter=args.max_iter)

    # --- compute tip-pose error ------------------------------------------
    T_hat = fwd_frame(m_hat, 1.0, np.array([0,0,L_PHYS_MM]))
    p_hat_mm = T_hat[:3,3]                # mm
    p_gt_mm  = T_gt[-1,:3,3]*1000.0       # convert m → mm

    pos_err = norm(p_hat_mm - p_gt_mm)
    R_hat = T_hat[:3,:3]
    R_gt  = T_gt[-1,:3,:3]
    cosang = np.clip((np.trace(R_gt.T@R_hat)-1)/2, -1, 1)
    rot_err = np.degrees(np.arccos(cosang))

    print(f"Sample #{args.idx}")
    print(f"  iterations      : {len(innov_log)}")
    print(f"  final innov norm: {innov_log[-1]:.2e}")
    print(f"  position error  : {pos_err:.3f} mm")
    print(f"  orientation err : {rot_err:.3f} deg")
    print(m_hat)
    # ----- visualise convergence -----------------------------------------
    plt.figure(figsize=(4,3))
    plt.semilogy(innov_log, '-o')
    plt.xlabel('IEKF iteration')
    plt.ylabel('innovation norm')
    plt.title(f"Convergence for sample {args.idx}")
    plt.grid(True)
    plt.tight_layout(); plt.show()


    ipdb.set_trace()

    # ---------- plot -----------------------------------------------------------
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p_gt_mm[0], p_gt_mm[1], p_gt_mm[2], 'k-', lw=2, label='Ground truth')
    ax.plot(p_hat_mm[0], p_hat_mm[1], p_hat_mm[2], 'r--', lw=2, label='EKF estimate')


    ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]'); ax.set_zlabel('Z [mm]')
    ax.set_title('Single sample backbone - GT vs. EKF')
    ax.legend(loc='upper left')
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()

if __name__ == "__main__":
    main()
