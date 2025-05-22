#!/usr/bin/env python3

import numpy as np
import math
import time
from numpy.linalg import norm, pinv
from scipy.integrate import solve_ivp, quad
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

##############################################################################
# 0) Utility to compute final tip error
##############################################################################
def tip_error(frames, x_des, z_des, theta_des):
    """
    Returns [dx, dz, dtheta] between the last frame in 'frames' and
    the desired tip pose (x_des, z_des, theta_des).
    
    frames.shape = (4, 4*N). Each 4x4 block is a pose along the rod.
    The final block is frames[:, -4:].
    """
    T_end = frames[:, -4:]
    x_end = T_end[0, 3]
    z_end = T_end[2, 3]
    # Orientation around Y => theta = atan2(T[0,2], T[2,2])
    theta_end = math.atan2(T_end[0, 2], T_end[2, 2])
    return np.array([x_end - x_des, z_end - z_des, theta_end - theta_des])


##############################################################################
# 1) Integration of curvature => frames
##############################################################################
def build_inplane_frames(kappa_func, length, number_disks=10):
    """
    Integrate curvature from s=0..length, subdividing each 'disk' into 10 substeps.
    Returns frames of shape (4, 4*(number_disks+1)).
    """
    frames = np.zeros((4, 4*(number_disks+1)))
    frames[:, 0:4] = np.eye(4)

    def theta_of_s_local(s):
        val, _ = quad(kappa_func, 0, s)
        return val

    ds = length / number_disks
    x_cur = 0.0
    z_cur = 0.0
    th_cur= 0.0
    s_current = 0.0

    for i in range(number_disks+1):
        # Build 4x4 pose
        R_3x3 = np.array([
            [ math.cos(th_cur), 0, math.sin(th_cur) ],
            [ 0,                1,               0  ],
            [-math.sin(th_cur), 0, math.cos(th_cur) ]
        ])
        T = np.eye(4)
        T[0:3, 0:3] = R_3x3
        T[0, 3] = x_cur
        T[2, 3] = z_cur
        frames[:, 4*i : 4*(i+1)] = T

        if i == number_disks:
            break

        # Subdivide for better geometry
        n_substep = 10
        local_ds = ds / n_substep
        for _ in range(n_substep):
            s_mid  = s_current + 0.5*local_ds
            th_mid = theta_of_s_local(s_mid)
            dx = math.sin(th_mid)*local_ds
            dz = math.cos(th_mid)*local_ds
            x_cur += dx
            z_cur += dz
            s_current += local_ds
        th_cur = theta_of_s_local(s_current)

    return frames


##############################################################################
# 2) PCK0 approach (constant curvature)
##############################################################################
def hp_0(L, l, X):
    """
    For PCK0, X = [m0].
    theta(s) = m0*s, s in [0..(l/L)].
    """
    s_val = l / L
    m0 = X[0]
    def theta_of_s(u): return m0*u
    def f_sin(u): return math.sin(theta_of_s(u))
    def f_cos(u): return math.cos(theta_of_s(u))
    x_int, _ = quad(f_sin, 0, s_val)
    z_int, _ = quad(f_cos, 0, s_val)
    x_tip = L*x_int
    z_tip = L*z_int
    tip_theta = theta_of_s(s_val)
    return np.array([x_tip, z_tip, tip_theta])

def build_inplane_polynomial_frames_0th(m0, length=0.1, number_disks=10):
    """
    Build frames from constant curvature 'm0'.
    """
    def kappa_func(s):
        return m0
    return build_inplane_frames(kappa_func, length, number_disks)

def pck0_inverse_2d_ls(x_tip, z_tip, theta_tip,
                       length=0.1, number_disks=10,
                       solver_tol=1e-6):
    """
    Solve for [m0] so final tip = (x_tip, z_tip, theta_tip).
    """
    def residual_fn(X):
        pose = hp_0(length, length, X)
        return pose - np.array([x_tip, z_tip, theta_tip])
    x0 = np.array([1.0])
    sol = least_squares(residual_fn, x0, method='lm',
                        ftol=solver_tol, xtol=solver_tol, gtol=solver_tol)
    m0_sol = sol.x[0]
    frames = build_inplane_polynomial_frames_0th(m0_sol, length, number_disks)
    return frames


##############################################################################
# 3) PCK2 approach (2nd-order polynomial curvature)
##############################################################################
def theta_of_s_pck2(s, m0, m1, m2):
    return m0*s + 0.5*m1*(s**2) + (m2*(s**3))/3.0

def hp_2(L, l, X):
    """
    Integrate from s=0..(l/L). X=[m0, m1, m2].
    """
    s_val = l / L
    m0, m1, m2 = X
    def theta_of_s(u):
        return m0*u + 0.5*m1*(u**2) + (m2*(u**3))/3.0
    def f_sin(u): return math.sin(theta_of_s(u))
    def f_cos(u): return math.cos(theta_of_s(u))
    x_int, _ = quad(f_sin, 0, s_val)
    z_int, _ = quad(f_cos, 0, s_val)
    x_tip = L*x_int
    z_tip = L*z_int
    tip_theta = theta_of_s(s_val)
    return np.array([x_tip, z_tip, tip_theta])

def build_inplane_polynomial_frames_2nd(X, length=0.1, number_disks=10):
    """
    Build frames from X=[m0,m1,m2].
    kappa(s)=m0 + m1*(s/L) + m2*(s/L)^2
    """
    m0, m1, m2 = X
    def kappa_func(s):
        s_norm = s/length
        return m0 + m1*s_norm + m2*(s_norm**2)
    return build_inplane_frames(kappa_func, length, number_disks)

def pck2_inverse_2d_ls(x_tip, z_tip, theta_tip,
                       length=0.1, number_disks=10,
                       solver_tol=1e-6):
    """
    Solve for [m0,m1,m2].
    """
    def residual_fn(X):
        pose = hp_2(length, length, X)
        return pose - np.array([x_tip, z_tip, theta_tip])
    x0 = np.array([1.0, 2.0, 3.0])
    sol = least_squares(residual_fn, x0, method='lm',
                        ftol=solver_tol, xtol=solver_tol, gtol=solver_tol)
    X_sol = sol.x
    frames = build_inplane_polynomial_frames_2nd(X_sol, length, number_disks)
    return frames


##############################################################################
# 4) Cosserat rod (CRT) approach, uniform curvature
##############################################################################
def integrate_crt_inplane_no_load(u0, length, number_disks):
    """
    Integrate ODE with uniform curvature u0.
    State: [x,z,theta,s].
    dx/ds=sin(theta), dz/ds=cos(theta), dtheta/ds=u0.
    """
    def ode_fun(s, y):
        x_, z_, th_ = y[0], y[1], y[2]
        dx_ = math.sin(th_)
        dz_ = math.cos(th_)
        dth_= u0
        return [dx_, dz_, dth_, 1.0]

    ds = length / number_disks
    y0 = np.array([0.0, 0.0, 0.0, 0.0])  # [x,z,theta,s]
    s0 = 0.0
    states_list = [y0.copy()]

    for _ in range(number_disks):
        s1 = s0 + ds
        sol = solve_ivp(ode_fun, (s0, s1), y0, max_step=ds, rtol=1e-6, atol=1e-6)
        y_end = sol.y[:, -1]
        states_list.append(y_end.copy())
        y0 = y_end
        s0 = s1

    return np.array(states_list)

def build_frames_crt_inplane(states):
    """
    Convert states => frames, where states[i,:] = [x, z, theta, s].
    """
    n = states.shape[0]
    frames = np.zeros((4, 4*n))
    for i in range(n):
        x_, z_, th_ = states[i,0], states[i,1], states[i,2]
        R_3x3 = np.array([
            [ math.cos(th_), 0, math.sin(th_) ],
            [ 0,             1,           0   ],
            [-math.sin(th_), 0, math.cos(th_) ]
        ])
        T = np.eye(4)
        T[0:3,0:3] = R_3x3
        T[0,3] = x_
        T[2,3] = z_
        frames[:, 4*i : 4*(i+1)] = T
    return frames

def crt_inverse_2d_refined(x_tip, z_tip, theta_tip,
                           length=0.1, number_disks=10,
                           solver_tol=1e-6):
    """
    Solve for uniform curvature 'u0' so final tip = (x_tip,z_tip,theta_tip).
    """
    def boundary_res(u0_array):
        u0_val = float(u0_array[0])
        states = integrate_crt_inplane_no_load(u0_val, length, number_disks)
        x_end, z_end, th_end = states[-1,0], states[-1,1], states[-1,2]
        return [x_end - x_tip, z_end - z_tip, th_end - theta_tip]

    x0 = np.array([0.0])
    sol = least_squares(boundary_res, x0, method='lm',
                        ftol=solver_tol, xtol=solver_tol, gtol=solver_tol)
    u0_sol = float(sol.x[0])
    states = integrate_crt_inplane_no_load(u0_sol, length, number_disks)
    frames = build_frames_crt_inplane(states)
    return frames


##############################################################################
# 5) Compare the three approaches for shape only
##############################################################################
def compare_run(x_tip, z_tip, theta_tip, length=0.1, number_disks=10, tol=1e-6):
    """
    Compare:
      - PCK0 => pck0_inverse_2d_ls
      - PCK2 => pck2_inverse_2d_ls
      - CRT  => crt_inverse_2d_refined
    Measure times and final tip errors.
    """
    results = {}

    # PCK0
    t0 = time.perf_counter()
    frames_pck0 = pck0_inverse_2d_ls(x_tip, z_tip, theta_tip,
                                     length=length, number_disks=number_disks,
                                     solver_tol=tol)
    t1 = time.perf_counter()
    results["PCK0_time_ms"] = (t1 - t0)*1e3
    results["PCK0_err"]     = tip_error(frames_pck0, x_tip, z_tip, theta_tip)

    # PCK2
    t0 = time.perf_counter()
    frames_pck2 = pck2_inverse_2d_ls(x_tip, z_tip, theta_tip,
                                     length=length, number_disks=number_disks,
                                     solver_tol=tol)
    t1 = time.perf_counter()
    results["PCK2_time_ms"] = (t1 - t0)*1e3
    results["PCK2_err"]     = tip_error(frames_pck2, x_tip, z_tip, theta_tip)

    # CRT
    t0 = time.perf_counter()
    frames_crt = crt_inverse_2d_refined(x_tip, z_tip, theta_tip,
                                        length=length, number_disks=number_disks,
                                        solver_tol=tol)
    t1 = time.perf_counter()
    results["CRT_time_ms"] = (t1 - t0)*1e3
    results["CRT_err"]     = tip_error(frames_crt, x_tip, z_tip, theta_tip)

    return results


##############################################################################
# 6) Functions for external load calculation
##############################################################################
def integral_vec(func, a, b, N=100):
    """
    Numerically integrate a vector function func(s) over [a, b]
    using a simple composite trapezoid rule with N sub-intervals.
    """
    s_vals = np.linspace(a, b, N+1)
    h = (b - a)/N
    out_vec = np.zeros_like(func(a))
    for i, s in enumerate(s_vals):
        w = 0.5 if (i==0 or i==N) else 1.0
        out_vec += w * func(s)
    return h*out_vec

def compute_external_wrench(m0, m1, m2,
                            delta, E, I, L, r,
                            q1, tau, theta_meas):
    """
    Example external wrench calculation, as in the second script.
    Returns an estimated 6x1 vector [Fx,Fy,Fz, Mx,My,Mz].
    """
    # shape-based tip angle
    theta_e_calc = m0 + 0.5*m1 + (m2/3.0)

    # Build JpS by numeric integration
    def kappa(s):
        return m0 + m1*s + m2*(s**2)
    def theta_of_s(u):
        val, _ = quad(kappa, 0, u)
        return val
    def Rz_negdelta(v):
        c = math.cos(-delta)
        s = math.sin(-delta)
        R = np.array([[ c, -s,  0],
                      [ s,  c,  0],
                      [ 0,  0,  1]])
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

    # JpS col1 => wrt m0
    vec_m0 = L * integral_vec(lambda ss: partial_mi(ss, 0), 0, 1, N=100)
    JpS_col1 = Rz_negdelta(vec_m0)
    # JpS col2 => wrt m1
    vec_m1 = L * integral_vec(lambda ss: partial_mi(ss, 1), 0, 1, N=100)
    JpS_col2 = Rz_negdelta(vec_m1)
    # JpS col3 => wrt m2
    vec_m2 = L * integral_vec(lambda ss: partial_mi(ss, 2), 0, 1, N=100)
    JpS_col3 = Rz_negdelta(vec_m2)
    # JpS col4 => wrt delta
    vec_delta = L * integral_vec(partial_delta, 0, 1, N=100)
    JpS_col4 = Rz_negdelta(vec_delta)

    JpS = np.column_stack([JpS_col1, JpS_col2, JpS_col3, JpS_col4])

    # JwS (3x4) => simplified example
    c_th = math.cos(theta_e_calc)
    s_th = math.sin(theta_e_calc)
    c_dl = math.cos(delta)
    s_dl = math.sin(delta)
    JwS = np.array([
        [-c_th*s_dl,    -0.5*c_th*s_dl,    -(1.0/3.0)*c_th*s_dl,     s_th*c_dl],
        [ c_th*c_dl,     0.5*c_th*c_dl,     (1.0/3.0)*c_th*c_dl,     s_th*s_dl],
        [-s_th,         -0.5*s_th,         -(1.0/3.0)*s_th,          (c_th-1.0)]
    ])
    JxS = np.vstack([JpS, JwS])  # (6x4)

    # JqS (4x4) => placeholder
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

    # Elastic energy gradient (rough approximation)
    gradU_S = L*np.array([
        E*I*(m0 + 0.25*m1 + (1.0/3.0)*m2),
        E*I*(0.25*m0 + (1.0/3.0)*m1 + 0.125*m2),
        E*I*((1.0/3.0)*m0 + 0.125*m1 + 0.2*m2),
        0.0
    ])

    diffVec_4 = gradU_S - JqS.T.dot(tau)  # shape (4,)
    # pinv of JxS.T => shape (6,4)
    JxS_T_pinv = pinv(JxS.T)
    F_est = JxS_T_pinv.dot(diffVec_4)  # shape (6,)

    return F_est

# Versions that return just shape parameters for PCK0/2 or uniform curvature for CRT
def pck0_inverse_2d_ls_value(x_tip, z_tip, theta_tip, L=0.04, solver_tol=1e-6):
    def residual_fn(X):
        pose_est = hp_0(L, L, X)
        return pose_est - np.array([x_tip, z_tip, theta_tip])
    x0 = np.array([1.0])
    sol = least_squares(residual_fn, x0, method='lm',
                        ftol=solver_tol, xtol=solver_tol, gtol=solver_tol)
    return sol.x[0]

def pck2_inverse_2d_ls_value(x_tip, z_tip, theta_tip, L=0.04, solver_tol=1e-6):
    def residual_fn(X):
        pose_est = hp_2(L, L, X)
        return pose_est - np.array([x_tip, z_tip, theta_tip])
    x0 = np.array([1.0,2.0,3.0])
    sol = least_squares(residual_fn, x0, method='lm',
                        ftol=solver_tol, xtol=solver_tol, gtol=solver_tol)
    return sol.x

def crt_inverse_2d_value(x_tip, z_tip, theta_tip, length=0.04, number_disks=10, solver_tol=1e-6):
    def boundary_res(u0_array):
        u0_val = float(u0_array[0])
        def ode_fun(s, y):
            x_, z_, th_ = y
            return [math.sin(th_), math.cos(th_), u0_val]
        ds = length/number_disks
        y0 = [0.0, 0.0, 0.0]
        s0 = 0.0
        for _ in range(number_disks):
            s1 = s0 + ds
            sol = solve_ivp(ode_fun, (s0, s1), y0, max_step=ds, rtol=1e-9, atol=1e-9)
            y0 = sol.y[:, -1].tolist()
            s0 = s1
        x_end, z_end, th_end = y0
        return [x_end - x_tip, z_end - z_tip, th_end - theta_tip]
    x0 = np.array([0.0])
    sol = least_squares(boundary_res, x0, method='lm',
                        ftol=solver_tol, xtol=solver_tol, gtol=solver_tol)
    return float(sol.x[0])

def compare_load_estimation_times(x_tip, z_tip, theta_tip,
                                  L=0.04, r=0.0018, E=6.5e10, I=4.83e-15,
                                  delta_deg=45.0, q1=0.002,
                                  tau=None, number_disks=10,
                                  solver_tol=1e-6):
    if tau is None:
        tau = np.array([2.5, 2.5, 0.0, 0.0])

    delta = math.radians(delta_deg)

    # PCK0
    t0 = time.perf_counter()
    m0_val = pck0_inverse_2d_ls_value(x_tip, z_tip, theta_tip, L=L, solver_tol=solver_tol)
    F_cc = compute_external_wrench(m0_val, 0.0, 0.0, delta, E, I, L, r, q1, tau, theta_tip)
    t1 = time.perf_counter()
    time_pck0 = (t1 - t0)*1e3

    # PCK2
    t0 = time.perf_counter()
    m0p, m1p, m2p = pck2_inverse_2d_ls_value(x_tip, z_tip, theta_tip, L=L, solver_tol=solver_tol)
    F_poly = compute_external_wrench(m0p, m1p, m2p, delta, E, I, L, r, q1, tau, theta_tip)
    t1 = time.perf_counter()
    time_pck2 = (t1 - t0)*1e3

    # CRT
    t0 = time.perf_counter()
    u0_val = crt_inverse_2d_value(x_tip, z_tip, theta_tip, length=L, number_disks=number_disks, solver_tol=solver_tol)
    # For uniform curvature, treat it as (m0 = u0_val, m1=0,m2=0)
    F_crt = compute_external_wrench(u0_val, 0.0, 0.0, delta, E, I, L, r, q1, tau, theta_tip)
    t1 = time.perf_counter()
    time_crt = (t1 - t0)*1e3

    return {
        "PCK0_time_ms": time_pck0,
        "PCK2_time_ms": time_pck2,
        "CRT_time_ms":  time_crt,
        "F_pck0": F_cc,
        "F_pck2": F_poly,
        "F_crt":  F_crt
    }


##############################################################################
# MAIN
##############################################################################
if __name__ == "__main__":

    # ------------------------------------------------------------------------
    # PART A: Compare shape‐only solve times vs. number of disks
    # ------------------------------------------------------------------------
    L_robot = 0.04  # 40 mm
    # x_tip = 0.02
    # z_tip = 0.03
    # theta_deg = 30.0
    x_tip = 0.02 + 0.0005 * np.random.randn()       # ±0.5 mm variation
    z_tip = 0.03 + 0.0005 * np.random.randn()
    theta_deg = 30.0 + 1.0 * np.random.randn() 
    theta_tip = math.radians(theta_deg)

    # disks_list = [1,2,3,4,5,6,7,8,9,10,15,20]
    disks_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    N = 1  # repeat each test

    avg_pck0 = []
    std_pck0 = []
    avg_pck2 = []
    std_pck2 = []
    avg_crt  = []
    std_crt  = []

    for nd in disks_list:
        t_list_pck0 = []
        t_list_pck2 = []
        t_list_crt  = []
        for _ in range(N):
            res = compare_run(x_tip, z_tip, theta_tip, length=L_robot,
                              number_disks=nd, tol=1e-6)
            t_list_pck0.append(res["PCK0_time_ms"])
            t_list_pck2.append(res["PCK2_time_ms"])
            t_list_crt.append(res["CRT_time_ms"])
        avg_pck0.append(np.mean(t_list_pck0))
        std_pck0.append(np.std(t_list_pck0))
        avg_pck2.append(np.mean(t_list_pck2))
        std_pck2.append(np.std(t_list_pck2))
        avg_crt.append(np.mean(t_list_crt))
        std_crt.append(np.std(t_list_crt))

    # ------------------------------------------------------------------------
    # PART B: Compare shape+load solves vs. number of disks
    # ------------------------------------------------------------------------
    r_anchor = 0.0018
    E_mod    = 6.5e10
    I_sec    = 4.83e-15
    # tau_4    = np.array([2.17, 2.05, 0.0, 0.0])
    # q1_val   = 0.002
    # delta_deg = 45.0
    tau_4    = np.array([2.17, 2.05, 0.0, 0.0]) + 0.01 * np.random.randn(4)
    q1_val   = 0.002 + 0.00005 * np.random.randn()
    delta_deg = 45.0 + 1.0 * np.random.randn()       # ±1° variation

    load_pck0_means, load_pck0_stds = [], []
    load_pck2_means, load_pck2_stds = [], []
    load_crt_means,  load_crt_stds  = [], []

    for nd in disks_list:
        t_pck0_list = []
        t_pck2_list = []
        t_crt_list  = []
        for _ in range(N):
            t_res = compare_load_estimation_times(
                x_tip, z_tip, theta_tip,
                L=L_robot, r=r_anchor, E=E_mod, I=I_sec,
                delta_deg=delta_deg, q1=q1_val,
                tau=tau_4,
                number_disks=nd,
                solver_tol=1e-6
            )
            t_pck0_list.append(t_res["PCK0_time_ms"])
            t_pck2_list.append(t_res["PCK2_time_ms"])
            t_crt_list.append(t_res["CRT_time_ms"])
        load_pck0_means.append(np.mean(t_pck0_list))
        load_pck0_stds.append(np.std(t_pck0_list))
        load_pck2_means.append(np.mean(t_pck2_list))
        load_pck2_stds.append(np.std(t_pck2_list))
        load_crt_means.append(np.mean(t_crt_list))
        load_crt_stds.append(np.std(t_crt_list))

    # ------------------------------------------------------------------------
    # PLOT RESULTS
    # ------------------------------------------------------------------------
    # -- First plot: shape-only times
    plt.figure()  # Chart #1
    plt.errorbar(disks_list, avg_pck0, yerr=std_pck0, fmt='-o', label='PCK0')
    plt.errorbar(disks_list, avg_pck2, yerr=std_pck2, fmt='-s', label='PCK2')
    plt.errorbar(disks_list, avg_crt,  yerr=std_crt,  fmt='-^', label='CRT')
    plt.xlabel('Number of Disks')
    plt.ylabel('Computation Time (ms)')
    plt.title(f"Shape‐Only Solve Times (x={x_tip}, z={z_tip}, th={theta_deg} deg)")
    plt.grid(True)
    plt.legend()

    # -- Second plot: shape + load times
    plt.figure()  # Chart #2
    plt.errorbar(disks_list, load_pck0_means, yerr=load_pck0_stds, fmt='-o', label='PCK0 + load')
    plt.errorbar(disks_list, load_pck2_means, yerr=load_pck2_stds, fmt='-s', label='PCK2 + load')
    plt.errorbar(disks_list, load_crt_means,  yerr=load_crt_stds,  fmt='-^', label='CRT + load')
    plt.xlabel('Number of Disks (ODE steps in CRT, or integration steps for polynomials)')
    plt.ylabel('Computation Time (ms)')
    plt.title('Shape + External Load Estimation Times')
    plt.grid(True)
    plt.legend()

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    # Subplot #1
    ax1.errorbar(disks_list, avg_pck0, yerr=std_pck0, fmt='-o', label='PCK0(CC)')
    ax1.errorbar(disks_list, avg_pck2, yerr=std_pck2, fmt='-s', label='PCK2')
    ax1.errorbar(disks_list, avg_crt,  yerr=std_crt,  fmt='-^', label='CRT')
    ax1.set_xlabel('Number of Disks', fontsize=16)
    ax1.set_ylabel('Time (ms)', fontsize=16)
    ax1.set_title('Shape‐Only', fontsize=16)
    ax1.grid(True)
    ax1.legend()
    ax1.tick_params(axis='both', labelsize=15)     # tick labels font size
    ax1.legend(fontsize=15)  

    # Subplot #2
    ax2.errorbar(disks_list, load_pck0_means, yerr=load_pck0_stds, fmt='-o', label='PCK0(CC) + load')
    ax2.errorbar(disks_list, load_pck2_means, yerr=load_pck2_stds, fmt='-s', label='PCK2 + load')
    ax2.errorbar(disks_list, load_crt_means,  yerr=load_crt_stds, fmt='-^', label='CRT + load')
    ax2.set_xlabel('Number of Disks', fontsize=16)
    ax2.set_ylabel('Time (ms)', fontsize=16)
    ax2.set_title('Shape + External Load', fontsize=16)
    ax2.grid(True)
    ax2.legend()
    ax2.tick_params(axis='both', labelsize=15)     # tick labels font size
    ax2.legend(fontsize=15)  




    plt.tight_layout()
    plt.show()

