#!/usr/bin/env python3
"""
debug_four_samples.py  – run an IEKF on *multiple* TDCR samples
                         and visualise four example backbones
                         (GT vs. IEKF) in a 2×2 grid, including
                         orientation triads at the IMU positions.

Usage
-----
python debug_four_samples.py \
       --idxs 0,5,12,19 \
       --gt   tdcr_gt_samples_100.npz \
       --meas tdcr_meas_samples_100.npz \
       --max-iter 30
"""
from __future__ import annotations
import argparse, itertools
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm
import matplotlib.pyplot as plt

# ────────── EKF model parameters ───────────────────────────────────────────
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

# ────────── IEKF core ─────────────────────────────────────────────────────
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
            return m_new
        m_est, P_est = m_new, P_new

    return m_est  # return last iterate regardless

# ────────── plotting helpers ───────────────────────────────────────────────
def fwd_positions(m, s_dense):
    e3 = np.array([0,0,L_PHYS_MM])
    return np.array([fwd_frame(m, s, e3)[:3,3] for s in s_dense])

def plot_frame(ax, origin, Rm, color='k', scale=10.0):
    axes_col = ['r','g','b'] if color == 'k' else ['darkred','darkgreen','navy']
    for i in range(3):
        ax.quiver(origin[0], origin[1], origin[2],
                  Rm[0,i]*scale, Rm[1,i]*scale, Rm[2,i]*scale,
                  color=axes_col[i], linewidth=1.2, alpha=0.8)

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
    p.add_argument('--idxs', default='10,21,32,43',
                   help='comma-separated list of four sample indices')
    p.add_argument('--gt',   default='tdcr_gt_samples_100.npz')
    p.add_argument('--meas', default='tdcr_meas_samples_100.npz')
    p.add_argument('--max-iter', type=int, default=30)
    args = p.parse_args()

    # parse indices & load data ------------------------------------------------
    idxs = [int(s.strip()) for s in args.idxs.split(',')[:4]]
    if len(idxs) != 4:
        raise ValueError('Please provide *exactly* four indices.')
    gt   = np.load(args.gt,   allow_pickle=True)
    meas = np.load(args.meas, allow_pickle=True)

    # figure with 2×2 grid -----------------------------------------------------
    fig = plt.figure(figsize=(11,10))
    axes = fig.subplots(2, 2, subplot_kw={'projection':'3d'})
    axes = axes.flatten()

    s_dense = np.linspace(0, 1, 160)
    e3      = np.array([0,0,L_PHYS_MM])

    for ax, idx in zip(axes, idxs):
        T_gt   = gt['T'][idx]             # (n_disks,4,4)
        q_meas = meas['q_meas'][idx]      # (3,4)

        m_hat = run_iterated_EKF(q_meas, max_iter=args.max_iter)

        # backbones -----------------------------------------------------------
        pos_gt  = T_gt[:,:3,3] * 1000.0
        pos_est = fwd_positions(m_hat, s_dense)

        ax.plot(pos_gt[:,0],  pos_gt[:,1],  pos_gt[:,2],
                'k-', lw=2,  label='GT' if idx == idxs[0] else '')
        ax.plot(pos_est[:,0], pos_est[:,1], pos_est[:,2],
                'r--', lw=2, label='IEKF' if idx == idxs[0] else '')

        # IMU triads ----------------------------------------------------------
        for s in IMU_S:
            # ground-truth triad
            id_g = int(round(s*(len(T_gt)-1)))
            Tg   = T_gt[id_g]; Rg = Tg[:3,:3]; pg = Tg[:3,3]*1000.0
            plot_frame(ax, pg, Rg, color='k')
            # estimated triad
            Te = fwd_frame(m_hat, s, e3)
            plot_frame(ax, Te[:3,3], Te[:3,:3], color='r')

        ax.set_title(f'Sample #{idx}')
        ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]'); ax.set_zlabel('Z [mm]')
        ax.view_init(elev=20, azim=-60)
        set_equal_axis(ax)

    # one shared legend --------------------------------------------------------
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)

    plt.suptitle('Ground-truth vs. IEKF backbone shapes (4 examples)', fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

if __name__ == '__main__':
    main()
