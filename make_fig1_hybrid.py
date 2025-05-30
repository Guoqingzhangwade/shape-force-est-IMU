#!/usr/bin/env python3
"""
make_fig1_hybrid.py  –  run shape EKF on every sample and draw Fig. 1
                        (hybrid violin+box plot of tip position & orientation
                         error at convergence).

Author: Guoqing Zhang, 2025-05-21
"""
from __future__ import annotations
import argparse, json, datetime
import numpy as np
from numpy.linalg import inv, norm
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})
import ipdb

# ------------------------------------------------------------
# EKF model settings (same notation as your single-sample demo)
# ------------------------------------------------------------
STATE_DIM   = 6
IMU_S       = np.array([12.0/39, 25.0/39, 1.00])        # sensor positions (s/L)
L_PHYS_MM   = 100.0                             # physical length in mm
GAMMA       = 20                                # Magnus subdivisions
PROC_STD0   = 1e-5
PROC_STD1   = 1e-6
Q           = np.diag([PROC_STD0**2, PROC_STD1**2,
                      PROC_STD0**2, PROC_STD1**2,
                      PROC_STD0**2, PROC_STD1**2])
MEAS_STD_DEG = 1.0                              # as specified
R_SINGLE    = (np.deg2rad(MEAS_STD_DEG) ** 2) * np.eye(3)
N_ITER_EKF  = 10                              # iterations per sample
INIT_M      = np.zeros(STATE_DIM)               # start from straight

# --------------------- helpers (mostly copied) -----------------------------
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
    """Homogeneous frame at arclength s (piece-wise Magnus PoE)."""
    T = np.eye(4); h = s/gamma
    for k in range(1, gamma+1):
        T = T @ expm(magnus_psi(k*h, h, m, e3))
    T[:3, 3] *= 1
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
# ---------------------------------------------------------------------------

def run_single_EKF(q_meas_frame):
    """Return estimated modal coeffs after N_ITER_EKF iterations."""
    m_est = INIT_M.copy()
    P_est = 1e-1 * np.eye(STATE_DIM)
    for _ in range(N_ITER_EKF):
        # prediction (static: no state evolution)
        m_pred, P_pred = m_est, P_est + Q

        H_stack, r_stack = [], []
        for s, q_meas in zip(IMU_S, q_meas_frame):
            r = -theta(q_meas, m_pred, s, e3)           # residual
            H = jac_num(q_meas, m_pred, s, e3)
            H_stack.append(H)
            r_stack.append(r)
        H = np.vstack(H_stack)
        r = np.hstack(r_stack)
        R_big = np.kron(np.eye(len(IMU_S)), R_SINGLE)

        S = H @ P_pred @ H.T + R_big
        K = P_pred @ H.T @ inv(S)

        m_est = m_pred + K @ r
        P_est = (np.eye(STATE_DIM) - K @ H) @ P_pred
    return m_est

# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",   default="tdcr_gt_samples_100.npz")
    parser.add_argument("--meas", default="tdcr_meas_samples_100.npz")
    parser.add_argument("--fig",  default="fig1_pose_error_100.pdf")
    args = parser.parse_args()

    # ----- load data -------------------------------------------------------
    gt  = np.load(args.gt,   allow_pickle=True)
    meas= np.load(args.meas, allow_pickle=True)

    T_gt     = gt["T"]                        # (N, n_disks, 4,4)  ground truth
    q_meas   = meas["q_meas"]                 # (N, 3, 4)
    N, n_disks = T_gt.shape[:2]
    # ipdb.set_trace()
    # constants
    global e3
    e3 = np.array([0,0, L_PHYS_MM])           # twist translation vector

    pos_err = np.empty(N)
    rot_err = np.empty(N)

    # ----- EKF for each sample --------------------------------------------
    for k in range(N):
        if k == 19 or k == 38:
            continue
        m_hat = run_single_EKF(q_meas[k])     # 6-vector modal coeffs

        # predicted tip frame (s = 1 → L)
        T_hat = fwd_frame(m_hat, 1.0, e3)     # returns mm translation

        # ground truth tip frame (convert metres → mm)
        T_true = T_gt[k, -1]                  # 4×4
        p_gt_mm = T_true[:3, 3] * 1000.0
        R_gt    = T_true[:3, :3]

        p_hat = T_hat[:3, 3]
        R_hat = T_hat[:3, :3]

        pos_err[k] = norm(p_hat - p_gt_mm)    # mm
        # if pos_err[k] > 40:
        #     print(k)
        cosang = (np.trace(R_gt.T @ R_hat) - 1)/2
        cosang = np.clip(cosang, -1.0, 1.0)
        rot_err[k] = np.degrees(np.arccos(cosang))

    # save quick summary
    print(f"[{datetime.datetime.now().time().isoformat(timespec='seconds')}] "
          f"Computed errors for {N} samples:")
    print(f"  Position error  mean ± σ : {pos_err.mean():.3f} ± {pos_err.std():.3f} mm")
    print(f"  Orientation err mean ± σ : {rot_err.mean():.3f} ± {rot_err.std():.3f} deg")

    # ----- hybrid violin + box plot ---------------------------------------
    fig, ax = plt.subplots(figsize=(3.5,5))

    parts = ax.violinplot([pos_err, rot_err],
                          positions=[1,2],
                          showmeans=False, showextrema=False,
                          widths=0.7)

    # Violin colour
    for pc in parts['bodies']:
        pc.set_facecolor('#7293CB')
        pc.set_alpha(0.65)

    # overlay box + mean
    bp = ax.boxplot([pos_err, rot_err],
                    positions=[1,2],
                    widths=0.15,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='black',
                                   markersize=5),
                    patch_artist=True)

    for patch in bp['boxes']:
        patch.set(facecolor='white', alpha=1.0)

    ax.set_xticks([1,2])
    ax.set_xticklabels(['Position\n(mm)', 'Orientation\n(deg)'])
    ax.set_ylabel('Error at convergence')
    ax.set_title('Fig. 1  Shape-EKF tip-pose error (N = {:d})'.format(N))
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(args.fig)
    plt.show()
    print(f"Figure saved ➜ {args.fig}")

if __name__ == "__main__":
    main()
