#!/usr/bin/env python3
"""
debug_covariance_ellipsoids.py  –  IEKF on one TDCR sample with
                                   full 1-σ positional ellipsoids
                                   rendered at each disk.

The figure shows
  • Ground-truth vs. mean IEKF backbone (left, 3-D)
  • A translucent 1-σ ellipsoid at every (skip-)disk centre
  • √λₘₐₓ(Σ_pos) vs. arclength (right, 2-D)

Usage
-----
python debug_covariance_ellipsoids.py        \
       --idx 25                              \
       --gt tdcr_gt_samples_100.npz          \
       --meas tdcr_meas_samples_100.npz      \
       --skip 2         # plot every 2nd disk
"""
from __future__ import annotations
import argparse, numpy as np
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 – keeps 3-D working
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm

# ───── EKF & robot constants (identical to previous script) ────────────────
STATE_DIM    = 6
IMU_S        = np.array([12.0/39, 25.0/39, 1.00])
L_PHYS_MM    = 100.0
GAMMA        = 20
PROC_STD0, PROC_STD1 = 1e-5, 1e-6
Q = np.diag([PROC_STD0**2, PROC_STD1**2]*3)
MEAS_STD_DEG = 1.0
R_SINGLE     = (np.deg2rad(MEAS_STD_DEG)**2) * np.eye(3)
INIT_M       = np.zeros(STATE_DIM)

# ───── helper kinematics (unchanged) ───────────────────────────────────────
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
    T = np.eye(4); h = s/gamma
    for k in range(1, gamma+1):
        T = T @ expm(magnus_psi(k*h, h, m, e3))
    return T

# ───── IEKF (returns m̂, P̂) ───────────────────────────────────────────────
def run_iterated_EKF(q_meas_frame, max_iter=30, tol=1e-6):
    e3 = np.array([0,0,L_PHYS_MM])
    m_est = INIT_M.copy(); P_est = 1e-1*np.eye(STATE_DIM)
    R_big = np.kron(np.eye(len(IMU_S)), R_SINGLE)

    def theta(q_meas, m, s):
        q_pred = R.from_matrix(fwd_frame(m, s, e3)[:3,:3]).as_quat()  # xyzw
        q_pred = np.roll(q_pred, 1)                                   # wxyz
        if q_pred[0] < 0: q_pred = -q_pred
        q_e = quat_mul(q_meas, quat_inv(q_pred))
        if q_e[0] < 0: q_e = -q_e
        return 2*q_e[1:]

    def jac_num(q_meas, m, s, eps=1e-6):
        J = np.zeros((3, STATE_DIM))
        for i in range(STATE_DIM):
            m_p, m_m = m.copy(), m.copy()
            m_p[i] += eps; m_m[i] -= eps
            J[:, i] = (theta(q_meas, m_p, s) -
                       theta(q_meas, m_m, s)) / (2*eps)
        return J

    for _ in range(max_iter):
        m_pred, P_pred = m_est, P_est + Q            # prediction (static model)

        H_blocks, r_blocks = [], []
        for s, q_meas in zip(IMU_S, q_meas_frame):
            r = -theta(q_meas, m_pred, s)
            H = jac_num(q_meas, m_pred, s)
            H_blocks.append(H); r_blocks.append(r)
        H = np.vstack(H_blocks); r = np.hstack(r_blocks)

        S = H @ P_pred @ H.T + R_big
        K = P_pred @ H.T @ np.linalg.inv(S)
        m_new = m_pred + K @ r
        P_new = (np.eye(STATE_DIM) - K @ H) @ P_pred

        if norm(m_new - m_est) < tol:
            return m_new, P_new
        m_est, P_est = m_new, P_new

    return m_est, P_est

# quaternion helpers --------------------------------------------------------
def quat_mul(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def quat_inv(q):
    w,x,y,z = q
    return np.array([w,-x,-y,-z]) / np.dot(q,q)

# ───── Jacobian of position wrt modal coeffs (FD) ─────────────────────────
def jac_pos_num(m, s, e3, eps=1e-6):
    J = np.zeros((3, STATE_DIM))
    for i in range(STATE_DIM):
        m_p, m_m = m.copy(), m.copy()
        m_p[i] += eps; m_m[i] -= eps
        J[:, i] = (fwd_frame(m_p, s, e3)[:3,3] -
                   fwd_frame(m_m, s, e3)[:3,3]) / (2*eps)
    return J

# ───── propagate Σ_pos to each disk ───────────────────────────────────────
def pos_covariances(m, P, s_list):
    e3 = np.array([0,0,L_PHYS_MM])
    Σs = []
    for s in s_list:
        J = jac_pos_num(m, s, e3)
        Σs.append(J @ P @ J.T)
    return Σs

# ───── draw an ellipsoid in world coords ──────────────────────────────────
def draw_cov_ellipsoid(ax, centre, Σ, n_std=1., color='dodgerblue',
                       alpha=0.15, wire=False):
    vals, vecs = eig(Σ)
    # numerical noise → keep real parts
    vals = np.real(vals); vecs = np.real(vecs)
    radii = n_std * np.sqrt(vals.clip(min=0))
    # parametric sphere
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    pts = np.stack((x, y, z), axis=-1)      # [...,3]
    # scale & rotate
    pts = pts * radii @ vecs.T              # broadcast radii
    # translate
    pts = pts + centre
    if wire:
        ax.plot_wireframe(pts[...,0], pts[...,1], pts[...,2],
                          rstride=2, cstride=2, color=color, alpha=alpha)
    else:
        ax.plot_surface(pts[...,0], pts[...,1], pts[...,2],
                        rstride=1, cstride=1, color=color, alpha=alpha,
                        linewidth=0)

# ───── utilities ──────────────────────────────────────────────────────────
def set_equal_axis(ax):
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    spans = [xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]]
    max_span = max(spans)
    ctrs = [0.5*(lim[0]+lim[1]) for lim in (xlim, ylim, zlim)]
    ax.set_xlim3d([ctrs[0]-max_span/2, ctrs[0]+max_span/2])
    ax.set_ylim3d([ctrs[1]-max_span/2, ctrs[1]+max_span/2])
    ax.set_zlim3d([ctrs[2]-max_span/2, ctrs[2]+max_span/2])

# ───── main ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, default=25)
    ap.add_argument('--gt',   default='tdcr_gt_samples_100.npz')
    ap.add_argument('--meas', default='tdcr_meas_samples_100.npz')
    ap.add_argument('--max-iter', type=int, default=50)
    ap.add_argument('--skip', type=int, default=1,
                    help='plot every <skip>-th disk ellipsoid')
    args = ap.parse_args()

    # --- data --------------------------------------------------------------
    gt   = np.load(args.gt,   allow_pickle=True)
    meas = np.load(args.meas, allow_pickle=True)
    T_gt   = gt['T'][args.idx]          # (n_disks,4,4)
    q_meas = meas['q_meas'][args.idx]   # (3,4) in wxyz already?

    # --- IEKF --------------------------------------------------------------
    m_hat, P_hat = run_iterated_EKF(q_meas, max_iter=args.max_iter)

    # arclength for each disk
    n_disks = T_gt.shape[0]
    s_disks = np.linspace(0, 1, n_disks)

    # propagate Σ_pos & mean backbone
    Σ_disks = pos_covariances(m_hat, P_hat, s_disks)
    pos_mean = np.array([fwd_frame(m_hat, s, np.array([0,0,L_PHYS_MM]))[:3,3]
                         for s in s_disks])
    pos_gt   = T_gt[:,:3,3] * 1000.0

    # scalar spread for right-hand plot
    sigmas = np.sqrt([np.max(np.real(eig(Σ)[0])) for Σ in Σ_disks])

    # --- figure ------------------------------------------------------------
    fig = plt.figure(figsize=(13,6))
    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)

    # backbones
    ax3d.plot(*pos_gt.T, 'k-', lw=2, label='Ground truth')
    ax3d.plot(*pos_mean.T, 'r--', lw=2, label='IEKF mean')

    # ellipsoids
    for i in range(0, n_disks, args.skip):
        draw_cov_ellipsoid(ax3d, pos_mean[i], Σ_disks[i],
                           n_std=30, color='royalblue', alpha=0.18, wire=True)

    ax3d.set_title('Backbone with 1 σ positional ellipsoids')
    ax3d.set_xlabel('X [mm]'); ax3d.set_ylabel('Y [mm]'); ax3d.set_zlabel('Z [mm]')
    ax3d.view_init(elev=20, azim=-60)
    set_equal_axis(ax3d)
    ax3d.legend(loc='upper left')

    # σ(s) plot
    ax2d.plot(s_disks*L_PHYS_MM, sigmas, 'b-')
    ax2d.set_xlabel('Arclength s [mm]')
    ax2d.set_ylabel('Positional σ [mm]')
    ax2d.set_title('√λₘₐₓ(Σ_pos) along the backbone')
    ax2d.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

