#!/usr/bin/env python3
"""
debug_covariance_impact.py  –  IEKF on a single TDCR sample *with*
                               uncertainty propagation along arclength.

The script plots:
  • GT vs. mean IEKF backbone in 3-D
  • ±1 σ uncertainty envelope in a normal-plane direction
  • √λₘₐₓ(Σ_pos(s)) as a function of arclength

Usage
-----
python debug_covariance_impact.py --idx 19 \
       --gt   tdcr_gt_samples_100.npz \
       --meas tdcr_meas_samples_100.npz \
       --max-iter 50
"""
from __future__ import annotations
import argparse, numpy as np
from numpy.linalg import norm, eigvals
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm

# ────────── EKF & robot constants ───────────────────────────────────────────
STATE_DIM    = 6
IMU_S        = np.array([12.0/39, 25.0/39, 1.00])   # arclength fractions
L_PHYS_MM    = 100.0
GAMMA        = 20
PROC_STD0, PROC_STD1 = 1e-5, 1e-6
Q = np.diag([PROC_STD0**2, PROC_STD1**2,
             PROC_STD0**2, PROC_STD1**2,
             PROC_STD0**2, PROC_STD1**2])
MEAS_STD_DEG = 1.0
R_SINGLE     = (np.deg2rad(MEAS_STD_DEG)**2) * np.eye(3)
INIT_M       = np.zeros(STATE_DIM)

# ────────── helper math ────────────────────────────────────────────────────
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
    k1, k2 = phi(c1) @ m, phi(c2) @ m
    e1, e2 = twist(k1, e3), twist(k2, e3)
    return (h/2)*(e1+e2) + (np.sqrt(3)/12)*h**2*(e1@e2 - e2@e1)

def fwd_frame(m, s, e3, gamma=GAMMA):
    """4×4 pose at arclength *s* (translation in mm)."""
    T = np.eye(4); h = s/gamma
    for k in range(1, gamma+1):
        T = T @ expm(magnus_psi(k*h, h, m, e3))
    return T

# quaternion utils ----------------------------------------------------------
def q_xyzw_to_wxyz(q):
    x, y, z, w = q
    return np.array([w, x, y, z])

def q_mul(a, b):
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
    q_pred = q_xyzw_to_wxyz(R.from_matrix(fwd_frame(m, s, e3)[:3,:3]).as_quat())
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

# ────────── IEKF returning covariance ─────────────────────────────────────
def run_iterated_EKF(q_meas_frame, max_iter=30, tol=1e-6):
    e3 = np.array([0,0,L_PHYS_MM])
    m_est = INIT_M.copy(); P_est = 1e-1*np.eye(STATE_DIM)
    R_big = np.kron(np.eye(len(IMU_S)), R_SINGLE)

    for _ in range(max_iter):
        m_pred, P_pred = m_est, P_est + Q  # static model prediction

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

        if norm(m_new - m_est) < tol:
            return m_new, P_new
        m_est, P_est = m_new, P_new

    return m_est, P_est

# ────────── uncertainty propagation helper ────────────────────────────────
def jac_pos_num(m, s, e3, eps=1e-6):
    J = np.zeros((3, STATE_DIM))
    for i in range(STATE_DIM):
        m_p, m_m = m.copy(), m.copy()
        m_p[i] += eps; m_m[i] -= eps
        pos_p = fwd_frame(m_p, s, e3)[:3,3]
        pos_m = fwd_frame(m_m, s, e3)[:3,3]
        J[:, i] = (pos_p - pos_m) / (2*eps)
    return J

def propagate_pos_cov(m, P, s_samples):
    """Returns list of 3×3 Σ_pos(s)."""
    e3 = np.array([0,0,L_PHYS_MM])
    covs = []
    for s in s_samples:
        J = jac_pos_num(m, s, e3)
        covs.append(J @ P @ J.T)
    return covs

# ────────── plotting utils ────────────────────────────────────────────────
def set_equal_axis(ax):
    xlims, ylims, zlims = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    spans = [xlims[1]-xlims[0], ylims[1]-ylims[0], zlims[1]-zlims[0]]
    max_span = max(spans)
    ctrs = [0.5*sum(lims) for lims in (xlims, ylims, zlims)]
    ax.set_xlim3d([ctrs[0]-max_span/2, ctrs[0]+max_span/2])
    ax.set_ylim3d([ctrs[1]-max_span/2, ctrs[1]+max_span/2])
    ax.set_zlim3d([ctrs[2]-max_span/2, ctrs[2]+max_span/2])

# ────────── main ───────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--idx', type=int, default=25)
    p.add_argument('--gt',   default='tdcr_gt_samples_100.npz')
    p.add_argument('--meas', default='tdcr_meas_samples_100.npz')
    p.add_argument('--max-iter', type=int, default=50)
    p.add_argument('--nsamp', type=int, default=160,
                   help='discretisation of arclength for covariance plot')
    args = p.parse_args()

    # load data -------------------------------------------------------------
    gt   = np.load(args.gt,   allow_pickle=True)
    meas = np.load(args.meas, allow_pickle=True)
    T_gt   = gt['T'][args.idx]          # (n_disks,4,4)
    q_meas = meas['q_meas'][args.idx]   # (3,4)

    # IEKF + covariance -----------------------------------------------------
    m_hat, P_hat = run_iterated_EKF(q_meas, max_iter=args.max_iter)
    print('Final modal-covariance diagonal:', np.diag(P_hat))

    # propagate Σ_pos(s) ----------------------------------------------------
    s_dense = np.linspace(0, 1, args.nsamp)
    cov_list = propagate_pos_cov(m_hat, P_hat, s_dense)
    sigmas   = np.sqrt([np.max(eigvals(S).real) for S in cov_list])  # 1-σ radius [mm]

    # mean backbone ---------------------------------------------------------
    pos_mean = np.array([fwd_frame(m_hat, s, np.array([0,0,L_PHYS_MM]))[:3,3]
                         for s in s_dense])
    pos_gt   = T_gt[:,:3,3] * 1000.0

    # ────────── figure -----------------------------------------------------
    fig = plt.figure(figsize=(13,5))
    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)

    # 3-D backbone + ±σ envelope (projection) ------------------------------
    ax3d.plot(pos_gt[:,0], pos_gt[:,1], pos_gt[:,2],
              'k-', lw=2, label='Ground truth')
    ax3d.plot(pos_mean[:,0], pos_mean[:,1], pos_mean[:,2],
              'r--', lw=2, label='IEKF mean')

    # draw ±σ whiskers every ~10 samples
    every = max(1, args.nsamp // 15)
    for k in range(0, args.nsamp, every):
        s      = s_dense[k]
        v_norm = np.array([0,0,1])                       # crude normal dir (z-axis)
        # if tangential dir is nearly z-axis, switch:
        T_s    = fwd_frame(m_hat, s, np.array([0,0,L_PHYS_MM]))
        t_hat  = T_s[:3,2] / norm(T_s[:3,2])
        if abs(np.dot(t_hat, v_norm)) > 0.9:
            v_norm = np.array([1,0,0])
        # orthogonalise & normalise
        v_norm = v_norm - np.dot(v_norm, t_hat)*t_hat
        v_norm = v_norm / norm(v_norm)

        p      = pos_mean[k]
        sigma  = sigmas[k]
        ax3d.plot([p[0]-sigma*v_norm[0], p[0]+sigma*v_norm[0]],
                  [p[1]-sigma*v_norm[1], p[1]+sigma*v_norm[1]],
                  [p[2]-sigma*v_norm[2], p[2]+sigma*v_norm[2]],
                  color='cornflowerblue', alpha=0.7, lw=1.4)

    ax3d.set_title('Backbone with ±1 σ positional envelope')
    ax3d.set_xlabel('X [mm]'); ax3d.set_ylabel('Y [mm]'); ax3d.set_zlabel('Z [mm]')
    ax3d.view_init(elev=20, azim=-60)
    set_equal_axis(ax3d)
    ax3d.legend(loc='upper left')

    # 2-D σ(s) plot --------------------------------------------------------
    ax2d.plot(s_dense*L_PHYS_MM, sigmas, 'b-')
    ax2d.set_xlabel('Arclength s [mm]')
    ax2d.set_ylabel('Positional σ [mm]')
    ax2d.set_title('√λ_max(Σ_pos) along the backbone')
    ax2d.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

