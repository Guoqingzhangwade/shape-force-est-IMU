# -*- coding: utf-8 -*-
"""
ekf_curvature_timeseries_quat.py
--------------------------------
Quaternion–driven EKF for 5-parameter curvature model using
analytic measurement Jacobian:
            H = J_{θq} · J_{qq} · J_{qR} · J_{Rm}
Only J_{Rm} (∂R/∂m) comes from a helper you already wrote.
Everything else is closed-form.
"""
# ───────────────────────── Imports ──────────────────────────
import os
import sys

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from shape_force_est_imu.ekf import (  # noqa: E402
    skew,
    q_fix_sign,
    forward_kinematics_multiple,
    ekf_update,
)

##############################################################################
# 5) Plot
##############################################################################

def plot_3d_curves(m_true, m_est, s_values, gamma=10, L=100.0):
    s_dense = np.linspace(0,1,100)
    rots_true, pos_true = forward_kinematics_multiple(m_true, s_dense, gamma=gamma, L=L)
    rots_est,  pos_est  = forward_kinematics_multiple(m_est,  s_dense, gamma=gamma, L=L)
    pos_true = np.array(pos_true)
    pos_est  = np.array(pos_est)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_true[:,0], pos_true[:,1], pos_true[:,2],
            color='blue', label='True shape')
    ax.plot(pos_est[:,0],  pos_est[:,1],  pos_est[:,2],
            color='green', linestyle='--', label='Est shape')

    rotsT, posT = forward_kinematics_multiple(m_true, s_values, gamma=gamma, L=L)
    rotsE, posE = forward_kinematics_multiple(m_est,  s_values, gamma=gamma, L=L)
    for (s_i, R_t, p_t, R_e, p_e) in zip(s_values, rotsT, posT, rotsE, posE):
        plot_frame(ax, p_t, R_t, f'True s={s_i}', 'blue')
        plot_frame(ax, p_e, R_e, f'Est s={s_i}', 'green')

    ax.set_xlabel('X(mm)')
    ax.set_ylabel('Y(mm)')
    ax.set_zlabel('Z(mm)')
    ax.legend()
    set_equal_axis(ax)
    plt.show()

def plot_frame(ax, origin, R, label=None, color='black'):
    scale = 5.0
    x_axis = R[:,0]*scale
    y_axis = R[:,1]*scale
    z_axis = R[:,2]*scale
    ax.quiver(origin[0], origin[1], origin[2],
              x_axis[0], x_axis[1], x_axis[2],
              color=color)
    ax.quiver(origin[0], origin[1], origin[2],
              y_axis[0], y_axis[1], y_axis[2],
              color=color)
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              color=color)
    if label:
        ax.text(origin[0], origin[1], origin[2], label, color=color)

def set_equal_axis(ax):
    xlims = ax.get_xlim3d()
    ylims = ax.get_ylim3d()
    zlims = ax.get_zlim3d()
    spans = [xlims[1]-xlims[0], ylims[1]-ylims[0], zlims[1]-zlims[0]]
    max_span = max(spans)
    x_center = 0.5*(xlims[0]+xlims[1])
    y_center = 0.5*(ylims[0]+ylims[1])
    z_center = 0.5*(zlims[0]+zlims[1])
    ax.set_xlim3d([x_center - max_span/2, x_center + max_span/2])
    ax.set_ylim3d([y_center - max_span/2, y_center + max_span/2])
    ax.set_zlim3d([z_center - max_span/2, z_center + max_span/2])


# ────────────────────────── Demo ────────────────────────────
if __name__ == "__main__":
    np.random.seed(44)

    # true parameters and virtual IMU positions
    m_true   = np.random.uniform(-4,4,5)
    s_vals   = [0.0, 0.3, 0.7, 1.0]
    L_phys   = 100.0
    gamma_int= 26
    σ_rot    = 0.005          # rad
    T_steps  = 500

    # ground-truth rotations
    R_true, _ = forward_kinematics_multiple(
        m_true, s_vals, gamma=gamma_int, L=L_phys, model="5d"
    )

    # generate noisy quaternion measurements
    meas_seq = []
    for _ in range(T_steps):
        q_list=[]
        for Rg in R_true:
            δw  = σ_rot*np.random.randn(3)
            Rn = expm(skew(δw)) @ Rg
            q   = R.from_matrix(Rn).as_quat()
            q = q_fix_sign(q)  # keep scalar part positive
            q_list.append(q)
        q_list[0] = np.array([0,0,0,1])  # base frame exactly known
        meas_seq.append(q_list)

    # EKF initialisation
    m_est = np.zeros(5)
    m_est = m_true
    P_est = np.eye(5)*0.1
    Q     = np.eye(5)*1e-6
    R_cov = np.eye(3*len(s_vals))*σ_rot**2

    m_hist = np.zeros((T_steps,5))
    for t in range(T_steps):
        # predict (random walk)
        P_pred = P_est + Q
        # update
        m_est, P_est = ekf_update(
            m_est,
            P_pred,
            s_vals,
            meas_seq[t],
            R_cov,
            gamma=gamma_int,
            L=L_phys,
            model="5d",
        )
        m_hist[t]=m_est

    # ─── plots ───────────────────────────────────────────────
    fig,axs = plt.subplots(5,1,sharex=True,figsize=(6,10))
    t = np.arange(T_steps)
    for i in range(5):
        axs[i].plot(t,m_hist[:,i],label=f'est m[{i}]')
        axs[i].axhline(m_true[i],ls='--',label='true')
        axs[i].legend(loc='upper right')
    axs[-1].set_xlabel('time step'); plt.tight_layout()

    # final shape vs true
    plot_3d_curves(m_true, m_est, s_vals, gamma=gamma_int, L=L_phys)
    # s_dense=np.linspace(0,1,100)
    # _,p_true=forward_kinematics_multiple(m_true,s_dense,gamma=gamma_int,L=L_phys)
    # _,p_est =forward_kinematics_multiple(m_est ,s_dense,gamma=gamma_int,L=L_phys)
    # p_true,p_est=np.asarray(p_true),np.asarray(p_est)

    # fig=plt.figure(); ax=fig.add_subplot(111,projection='3d')
    # ax.plot(*p_true.T,label='true'); ax.plot(*p_est.T,'--',label='est')
    # ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.legend()
    # plt.show()
