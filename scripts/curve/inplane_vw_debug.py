import argparse
import json
import math
import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from shape_force_est_imu.crt import virtual_work as vw


def Rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def grad_U_plane(S, E, I, L):
    m0 = S[0]
    m1 = S[1] if len(S) > 1 else 0.0
    m2 = S[2] if len(S) > 2 else 0.0
    c0 = (m0 + 0.5 * m1 + (1.0 / 3.0) * m2)
    c1 = (0.5 * m0 + (1.0 / 3.0) * m1 + 0.25 * m2)
    c2 = ((1.0 / 3.0) * m0 + 0.25 * m1 + 0.2 * m2)
    return (E * I / L) * np.array([c0, c1, c2], dtype=float)


def build_JqS_plane(S, Rp, delta):
    th_end = S[0] + 0.5 * (S[1] if len(S) > 1 else 0.0) + (1.0 / 3.0) * (
        S[2] if len(S) > 2 else 0.0
    )
    phi = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi], dtype=float)
    sig = phi + delta
    c = np.cos(sig)
    s = np.sin(sig)
    J = np.zeros((4, 4), dtype=float)
    J[:, 0] = Rp * c
    J[:, 1] = 0.5 * Rp * c
    J[:, 2] = (1.0 / 3.0) * Rp * c
    J[:, 3] = -Rp * th_end * s
    return J


def tendon_offsets(Rp, delta):
    phi = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi], dtype=float)
    sig = phi + delta
    return [np.array([Rp * math.cos(a), Rp * math.sin(a), 0.0]) for a in sig]

def cable_length_unit_s(m, r_i, order_x, order_y, order_z, n_int=400):
    s_vals = np.linspace(0.0, 1.0, n_int + 1)
    vals = np.zeros_like(s_vals)
    for idx, s in enumerate(s_vals):
        kappa = vw.curvature_from_modal(s, m, order_x, order_y, order_z)
        t_dir = np.array([0.0, 0.0, 1.0]) + vw.hat(kappa) @ r_i
        vals[idx] = np.linalg.norm(t_dir)
    return np.trapezoid(vals, s_vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m0", type=float, default=5.0)
    ap.add_argument("--m1", type=float, default=0.0)
    ap.add_argument("--m2", type=float, default=0.0)
    ap.add_argument("--L", type=float, default=0.10)
    ap.add_argument("--E", type=float, default=60e9)
    ap.add_argument("--r", type=float, default=5e-4)
    ap.add_argument("--Rp", type=float, default=0.008)
    ap.add_argument("--delta", type=float, default=math.pi / 4)
    ap.add_argument("--n-int", type=int, default=400)
    ap.add_argument("--order-x", type=int, default=1)
    ap.add_argument("--order-y", type=int, default=1)
    ap.add_argument("--order-z", type=int, default=0)
    ap.add_argument("--m-list", type=str,
                    default="[0.5, 0.1, -0.2, 0.05, 0.02]",
                    help="JSON list of modal coefficients for FD checks")
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="finite-difference step for Jacobian checks")
    ap.add_argument("--s-check", type=float, default=1.0,
                    help="arc-length location for body Jacobian check")
    ap.add_argument("--gamma", type=int, default=26,
                    help="Magnus subintervals for body Jacobian check")
    args = ap.parse_args()

    S = np.array([args.m0, args.m1, args.m2], dtype=float)
    I = math.pi * args.r ** 4 / 4.0

    grad_plane = grad_U_plane(S, args.E, I, args.L)
    Jq_plane = build_JqS_plane(S, args.Rp, args.delta)[:, :3]

    order_x, order_y, order_z = 0, 2, 0
    m = np.array([0.0, args.m0, args.m1, args.m2, 0.0], dtype=float)

    grad_vw = vw.elastic_energy_gradient(
        m,
        EIx=0.0,
        EIy=args.E * I,
        GJ=0.0,
        L=args.L,
        order_x=order_x,
        order_y=order_y,
        order_z=order_z,
    )
    grad_vw_y = grad_vw[1:4]

    r_list = tendon_offsets(args.Rp, args.delta)
    J_lm = vw.cable_jacobian(
        m,
        r_list,
        args.L,
        order_x=order_x,
        order_y=order_y,
        order_z=order_z,
        n_int=args.n_int,
    )
    J_lm_y = J_lm[:, 1:4]

    print("=== Gradient check (plane vs 3D helper) ===")
    print("grad_plane:", grad_plane)
    print("grad_vw_y :", grad_vw_y)
    print("diff      :", grad_vw_y - grad_plane)
    print("")
    print("=== Cable Jacobian check (plane vs 3D helper) ===")
    print("Jq_plane (m0..m2):")
    print(Jq_plane)
    print("J_lm_y (m0..m2):")
    print(J_lm_y)
    print("diff:")
    print(J_lm_y - Jq_plane)

    try:
        m_fd = np.array(json.loads(args.m_list), dtype=float)
    except Exception as exc:
        raise ValueError(f"Failed to parse --m-list JSON: {exc}")
    n_params = vw.total_params(args.order_x, args.order_y, args.order_z)
    if m_fd.size != n_params:
        raise ValueError(f"--m-list length {m_fd.size} != {n_params} params")

    angles = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    r_list = [np.array([args.Rp * np.cos(a), args.Rp * np.sin(a), 0.0]) for a in angles]

    J_lm = vw.cable_jacobian(
        m_fd, r_list, args.L,
        order_x=args.order_x, order_y=args.order_y, order_z=args.order_z,
        n_int=args.n_int,
    )
    J_lm_fd = np.zeros_like(J_lm)
    for i_param in range(n_params):
        dm = np.zeros(n_params)
        dm[i_param] = args.eps
        m_p = m_fd + dm
        m_m = m_fd - dm
        for i_r, r_i in enumerate(r_list):
            Lp = cable_length_unit_s(m_p, r_i, args.order_x, args.order_y, args.order_z, args.n_int)
            Lm = cable_length_unit_s(m_m, r_i, args.order_x, args.order_y, args.order_z, args.n_int)
            J_lm_fd[i_r, i_param] = -(Lp - Lm) / (2.0 * args.eps)

    print("")
    print("=== Finite-diff check: cable_jacobian ===")
    diff = J_lm - J_lm_fd
    max_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    max_abs = float(np.max(np.abs(diff)))
    denom = max(1.0, float(np.max(np.abs(J_lm_fd))))
    print("max abs diff:", max_abs)
    print("max rel diff:", max_abs / denom)
    print("worst idx:", max_idx, "analytic:", J_lm[max_idx], "fd:", J_lm_fd[max_idx])

    J_vb, T_base = vw.body_jacobian_at_s(
        m_fd, args.s_check, gamma=args.gamma, L=args.L,
        order_x=args.order_x, order_y=args.order_y, order_z=args.order_z
    )
    J_vb_fd = np.zeros_like(J_vb)
    for i_param in range(n_params):
        dm = np.zeros(n_params)
        dm[i_param] = args.eps
        T_p = vw.product_of_exponentials(
            m_fd + dm, args.s_check, args.gamma, args.L,
            args.order_x, args.order_y, args.order_z
        )
        T_m = vw.product_of_exponentials(
            m_fd - dm, args.s_check, args.gamma, args.L,
            args.order_x, args.order_y, args.order_z
        )
        dT = (T_p - T_m) / (2.0 * args.eps)
        Vb_hat = np.linalg.inv(T_base) @ dT
        J_vb_fd[:, i_param] = vw.vee(Vb_hat)

    print("=== Finite-diff check: body_jacobian_at_s ===")
    diff = J_vb - J_vb_fd
    max_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    max_abs = float(np.max(np.abs(diff)))
    denom = max(1.0, float(np.max(np.abs(J_vb_fd))))
    print("max abs diff:", max_abs)
    print("max rel diff:", max_abs / denom)
    print("worst idx:", max_idx, "analytic:", J_vb[max_idx], "fd:", J_vb_fd[max_idx])


if __name__ == "__main__":
    main()
