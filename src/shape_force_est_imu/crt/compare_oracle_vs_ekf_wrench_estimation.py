#!/usr/bin/env python3
"""
Oracle vs EKF Wrench Estimation Comparison.

Answers: Is the 55% force NRMSE due to bad shape estimation (EKF shape error)
or fundamental observability limits (virtual-work null space)?

Two methods, tested over multiple modal orders:

  A. Oracle-fit  — fit polynomial modal coefficients DIRECTLY to GT backbone
                   frames (perfect shape, no noise), then compute wrench via
                   virtual-work equations.  Upper bound on wrench accuracy.

  B. EKF-based   — run the full SO(3)-analytic EKF on noisy IMU measurements,
                   obtain m_est, then compute wrench via virtual-work equations.
                   Same pipeline as evaluate_kirchhoff_wrench_estimation.py.

For both, only the Direct (pseudo-inverse) wrench solver is used so that shape
estimation error is the only variable between A and B.  MAP is omitted here
to avoid confounding factors.

Modal orders tested
-------------------
  (1, 1, 0)  — current 5-D EKF state  [k0x, k1x, k0y, k1y, k0z]
  (1, 2, 0)  — add k2y  → 6-D
  (2, 2, 0)  — add k2x  → 7-D
  (2, 2, 1)  — add k1z  → 8-D

Output (in --save-dir)
----------------------
  oracle_vs_ekf_results.csv        per-(case, order, layout, method)
  oracle_vs_ekf_summary.csv        aggregated means
  oracle_vs_ekf_results.json       full structured results
  oracle_vs_ekf_force.pdf/.png     force error violin per order × method
  oracle_vs_ekf_moment.pdf/.png    moment error violin per order × method
  oracle_vs_ekf_bars.pdf/.png      bar chart: oracle vs EKF per order

Usage
-----
  cd src/shape_force_est_imu/crt
  python compare_oracle_vs_ekf_wrench_estimation.py
  python compare_oracle_vs_ekf_wrench_estimation.py \\
      --gt gt_data/kirchhoff_gt_dataset.npz \\
      --imu2 gt_data/kirchhoff_imu_2imu.npz \\
      --save-dir gt_data/results
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import block_diag as _block_diag
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# EKF utilities re-used from Step 3
# ---------------------------------------------------------------------------
from evaluate_kirchhoff_shape_estimation import (
    ALPHA_DEFAULT,
    E3,
    GAMMA_DEFAULT,
    MEAS_STD_DEG_DEF,
    P0_DIAG,
    Q_DIAG,
    STEPS_DEFAULT,
    load_ground_truth_dataset,
    load_imu_measurements,
    run_ekf_on_frame,
)

# ---------------------------------------------------------------------------
# Virtual-work / mechanics
# ---------------------------------------------------------------------------
from virtual_work import (
    body_jacobian_at_s,
    elastic_energy_gradient,
    generalized_modal_load,
    gram_matrix,
    pull_jacobian,
    solve_wrench,
)

# ---------------------------------------------------------------------------
# Rod constants (match generate_kirchhoff_gt_dataset.py)
# ---------------------------------------------------------------------------
_L       = 0.1
_E       = 60e9
_NU      = 0.3
_G       = _E / (2 * (1 + _NU))
_R_BB    = 5e-4
_I_BB    = np.pi * _R_BB**4 / 4.0
_EIX     = _E * _I_BB
_EIY     = _EIX
_GJ      = _G * 2 * _I_BB
_R_TENDON = 0.008
_R_LIST = [
    np.array([_R_TENDON,  0.0,        0.0]),
    np.array([0.0,        _R_TENDON,  0.0]),
    np.array([-_R_TENDON, 0.0,        0.0]),
    np.array([0.0,       -_R_TENDON,  0.0]),
]

# ---------------------------------------------------------------------------
# Modal order configurations to test
# ---------------------------------------------------------------------------
class ModalOrder(NamedTuple):
    order_x: int
    order_y: int
    order_z: int
    label: str

MODAL_ORDERS = [
    ModalOrder(1, 1, 0, "(1,1,0)"),
    ModalOrder(1, 2, 0, "(1,2,0)"),
    ModalOrder(2, 2, 0, "(2,2,0)"),
    ModalOrder(2, 2, 1, "(2,2,1)"),
]

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
LABEL_FS  = 12
TICK_FS   = 10
TITLE_FS  = 12
LEGEND_FS = 10
ANN_FS    = 8.5
_METHOD_COLORS = {"oracle": "#2ca02c", "ekf": "#1f77b4"}
_METHOD_LABELS = {"oracle": "Oracle", "ekf": "EKF"}


# ---------------------------------------------------------------------------
# Elastic energy Hessian (per modal order)
# ---------------------------------------------------------------------------

def make_hessian(order_x: int, order_y: int, order_z: int) -> np.ndarray:
    Mx = gram_matrix(order_x)
    My = gram_matrix(order_y)
    Mz = gram_matrix(order_z)
    return _block_diag(
        (_EIX / _L) * Mx,
        (_EIY / _L) * My,
        (_GJ  / _L) * Mz,
    )


# ---------------------------------------------------------------------------
# Wrench from modal state (Direct, pseudo-inverse)
# ---------------------------------------------------------------------------

def wrench_from_modal(
    m: np.ndarray,
    tau: np.ndarray,
    order_x: int,
    order_y: int,
    order_z: int,
    gamma: int = GAMMA_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute world-frame (f, l) from modal state m and tendon tensions tau."""
    gradU = elastic_energy_gradient(m, _EIX, _EIY, _GJ, _L,
                                    order_x, order_y, order_z)
    J_qm = pull_jacobian(m, _R_LIST, _L, order_x, order_y, order_z)
    J_vbm, T_tip = body_jacobian_at_s(m, 1.0, gamma, _L,
                                       order_x, order_y, order_z)
    F_b = solve_wrench(J_vbm, J_qm, gradU, tau)
    R_tip = T_tip[:3, :3]
    f_world = R_tip @ F_b[3:]
    l_world = R_tip @ F_b[:3]
    return f_world, l_world


# ---------------------------------------------------------------------------
# Oracle fitting: GT backbone frames → modal coefficients (least squares)
# ---------------------------------------------------------------------------

def _rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to rotation vector (Rodrigues)."""
    return Rotation.from_matrix(R).as_rotvec()


def estimate_modal_oracle(
    positions_case: np.ndarray,
    orientations_case: np.ndarray,
    order_x: int,
    order_y: int,
    order_z: int,
    L_phys: float = _L,
) -> np.ndarray:
    """
    Fit polynomial modal coefficients to GT backbone frames.

    Uses SO(3) finite differences on consecutive frames to compute distributed
    curvature, then fits polynomial bases via linear least squares — exactly
    the same convention as virtual_work.py (normalised arc-length s ∈ [0,1]).

    Parameters
    ----------
    positions_case   : (M, 3)  GT backbone positions in metres (not used for R)
    orientations_case: (M, 9)  GT rotation matrices (row-major, flattened)

    Returns
    -------
    m_oracle : (n_params,) modal coordinates
    """
    M = len(positions_case)
    s_values = np.linspace(0.0, 1.0, M)
    ds = s_values[1] - s_values[0]
    s_mid = 0.5 * (s_values[:-1] + s_values[1:])

    kappa = np.zeros((M - 1, 3))
    for i in range(M - 1):
        R_i = orientations_case[i].reshape(3, 3)
        R_j = orientations_case[i + 1].reshape(3, 3)
        R_rel = R_i.T @ R_j
        rotvec = _rotation_matrix_to_rotvec(R_rel)
        kappa[i] = rotvec / ds

    def fit_axis(kappa_axis: np.ndarray, order: int) -> np.ndarray:
        Phi = np.vstack([s_mid ** p for p in range(order + 1)]).T
        return np.linalg.lstsq(Phi, kappa_axis, rcond=None)[0]

    mx = fit_axis(kappa[:, 0], order_x)
    my = fit_axis(kappa[:, 1], order_y)
    mz = fit_axis(kappa[:, 2], order_z)
    return np.hstack([mx, my, mz])


# ---------------------------------------------------------------------------
# General forward kinematics for arbitrary modal order
# ---------------------------------------------------------------------------

def _phi_for_order(s: float, order_x: int, order_y: int, order_z: int) -> np.ndarray:
    """Shape function matrix (3, n_params) for arbitrary (order_x, order_y, order_z)."""
    n_params = (order_x + 1) + (order_y + 1) + (order_z + 1)
    mat = np.zeros((3, n_params))
    ox1    = order_x + 1
    ox1oy1 = ox1 + (order_y + 1)
    for p in range(order_x + 1):
        mat[0, p] = s ** p
    for p in range(order_y + 1):
        mat[1, ox1 + p] = s ** p
    for p in range(order_z + 1):
        mat[2, ox1oy1 + p] = s ** p
    return mat


def fwd_transform_general(
    m: np.ndarray,
    s: float,
    gamma: int,
    L_phys: float,
    order_x: int,
    order_y: int,
    order_z: int,
) -> np.ndarray:
    """
    Full 4×4 SE(3) transform at normalised arc-length s for arbitrary modal order.
    T[:3,3] is position in physical metres (scaled by L_phys).
    """
    from scipy.linalg import expm as _expm
    from evaluate_kirchhoff_shape_estimation import skew as _skew, _GL_XI

    def _twist(k):
        S = _skew(k)
        T = np.zeros((4, 4))
        T[:3, :3] = S
        T[:3, 3] = E3
        return T

    def _psi(s_i, h):
        c1 = s_i - h + _GL_XI[0] * h
        c2 = s_i - h + _GL_XI[1] * h
        e1 = _twist(_phi_for_order(c1, order_x, order_y, order_z) @ m)
        e2 = _twist(_phi_for_order(c2, order_x, order_y, order_z) @ m)
        return (h / 2) * (e1 + e2) + (np.sqrt(3) / 12) * h**2 * (e1 @ e2 - e2 @ e1)

    T = np.eye(4)
    h = s / gamma
    for k in range(1, gamma + 1):
        T = T @ _expm(_psi(k * h, h))
    T[:3, 3] *= L_phys
    return T


# ---------------------------------------------------------------------------
# EKF: extract m_est for a given modal order
# ---------------------------------------------------------------------------

def run_ekf_for_order(
    R_frame: List[np.ndarray],
    imu_pos_norm: np.ndarray,
    order_x: int,
    order_y: int,
    order_z: int,
    gamma: int = GAMMA_DEFAULT,
    meas_std_deg: float = MEAS_STD_DEG_DEF,
    alpha: float = ALPHA_DEFAULT,
    steps: int = STEPS_DEFAULT,
) -> np.ndarray:
    """
    Run EKF for an arbitrary modal order configuration.

    The EKF is always initialised at m=0 and P0=diag([1,0.5,...]) scaled to
    n_params dimensions.  For orders beyond (1,1,0) the extra parameters use
    a modest initial uncertainty (0.25 rad/s^order, consistent with P0_DIAG).
    """
    n_params = (order_x + 1) + (order_y + 1) + (order_z + 1)

    # Scale P0 to n_params: reuse the last P0_DIAG entry for extra params
    base = list(P0_DIAG)
    while len(base) < n_params:
        base.append(0.25)
    P0_scaled = np.diag(base[:n_params])

    # Scale Q_DIAG similarly (process noise for extra params same as k1-terms)
    q_base = list(Q_DIAG)
    while len(q_base) < n_params:
        q_base.append(Q_DIAG[1])  # k1-level noise for higher-order terms
    Q_scaled = np.diag(q_base[:n_params])

    from evaluate_kirchhoff_shape_estimation import (
        so3_analytic_H_and_r, STATE_DIM
    )
    import importlib
    import evaluate_kirchhoff_shape_estimation as _ekf_mod
    from numpy.linalg import inv

    # --- Build a local EKF that handles arbitrary n_params ---
    sigma = np.deg2rad(meas_std_deg)
    R_sngl = alpha * sigma**2 * np.eye(3)
    n_imu = len(imu_pos_norm)
    R_big = np.kron(np.eye(n_imu), R_sngl)

    # Reuse the same SO(3) analytic H/r but with a custom phi_mat for n_params
    # We need to build the measurement Jacobian for general (ox, oy, oz).
    from scipy.linalg import expm
    from evaluate_kirchhoff_shape_estimation import (
        skew, vee, so3_log, jr_inv_so3,
        _GL_XI, twist_mat, magnus_psi as _magnus_psi,
    )

    def phi_general(s: float) -> np.ndarray:
        """Shape function matrix for arbitrary modal order, shape (3, n_params).

        kappa(s) = phi_general(s) @ m  gives a 3-vector [kappa_x, kappa_y, kappa_z].
        """
        mat = np.zeros((3, n_params))
        ox1    = order_x + 1
        ox1oy1 = ox1 + (order_y + 1)
        for p in range(order_x + 1):
            mat[0, p] = s ** p
        for p in range(order_y + 1):
            mat[1, ox1 + p] = s ** p
        for p in range(order_z + 1):
            mat[2, ox1oy1 + p] = s ** p
        return mat  # (3, n_params)

    def twist_mat_local(k: np.ndarray, e3: np.ndarray) -> np.ndarray:
        return np.block([[skew(k), e3[:, None]],
                         [np.zeros((1, 3)), 0.0]])

    def magnus_psi_general(s_i: float, h: float, m: np.ndarray) -> np.ndarray:
        c1 = s_i - h + _GL_XI[0] * h
        c2 = s_i - h + _GL_XI[1] * h
        e1 = twist_mat_local(phi_general(c1) @ m, E3)
        e2 = twist_mat_local(phi_general(c2) @ m, E3)
        return (h / 2) * (e1 + e2) + (np.sqrt(3) / 12) * h**2 * (e1 @ e2 - e2 @ e1)

    def fwd_rotation_general(m: np.ndarray, s: float) -> np.ndarray:
        T = np.eye(4)
        h = s / gamma
        for k in range(1, gamma + 1):
            T = T @ expm(magnus_psi_general(k * h, h, m))
        return T[:3, :3]

    def compute_dR_dm_general(m: np.ndarray, s: float) -> np.ndarray:
        h = s / gamma
        Psi_list, ePsi_list = [], []
        T_before = [np.eye(4)]
        dPsi_dm  = [np.zeros((4, 4, n_params)) for _ in range(gamma)]
        dePsi_dm = [np.zeros((4, 4, n_params)) for _ in range(gamma)]

        for k in range(1, gamma + 1):
            Psi_k  = magnus_psi_general(k * h, h, m)
            ePsi_k = expm(Psi_k)
            Psi_list.append(Psi_k)
            ePsi_list.append(ePsi_k)
            T_before.append(T_before[-1] @ ePsi_k)

        T_after_arr = [np.eye(4) for _ in range(gamma + 1)]
        for k in range(gamma - 1, 0, -1):
            T_after_arr[k] = ePsi_list[k] @ T_after_arr[k + 1]

        for k in range(1, gamma + 1):
            s_i = k * h
            c1  = s_i - h + _GL_XI[0] * h
            c2  = s_i - h + _GL_XI[1] * h
            ph1 = phi_general(c1)
            ph2 = phi_general(c2)
            eta1 = twist_mat_local(ph1 @ m, E3)
            eta2 = twist_mat_local(ph2 @ m, E3)
            for i in range(n_params):
                e1_col = ph1[:, i]
                e2_col = ph2[:, i]
                de1 = twist_mat_local(e1_col, np.zeros(3))
                de2 = twist_mat_local(e2_col, np.zeros(3))
                comm = (de1 @ eta2 - eta2 @ de1) + (eta1 @ de2 - de2 @ eta1)
                dPsi_dm[k-1][:, :, i] = (
                    (h / 2) * (de1 + de2)
                    + (np.sqrt(3) / 12) * h**2 * comm
                )

        for k in range(gamma):
            ePsi_k = ePsi_list[k]
            Psi_k  = Psi_list[k]
            for i in range(n_params):
                dP = dPsi_dm[k][:, :, i]
                dePsi_dm[k][:, :, i] = ePsi_k @ (dP + 0.5 * (Psi_k @ dP - dP @ Psi_k))

        dT_dm = np.zeros((4, 4, n_params))
        for i in range(n_params):
            for k in range(gamma):
                dT_dm[:, :, i] += T_before[k] @ dePsi_dm[k][:, :, i] @ T_after_arr[k + 1]
        return dT_dm[:3, :3, :]

    def H_and_r_general(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H_rows, r_rows = [], []
        for s_i, R_meas in zip(imu_pos_norm, R_frame):
            R_pred = fwd_rotation_general(m, float(s_i))
            R_e    = R_meas @ R_pred.T
            r      = so3_log(R_e)
            Jr_inv = jr_inv_so3(r)
            dR_dm  = compute_dR_dm_general(m, float(s_i))
            H_i    = np.zeros((3, n_params))
            for p in range(n_params):
                A = dR_dm[:, :, p] @ R_pred.T
                H_i[:, p] = -Jr_inv @ vee(0.5 * (A - A.T))
            H_rows.append(H_i)
            r_rows.append(r)
        return np.vstack(H_rows), np.hstack(r_rows)

    m_est = np.zeros(n_params)
    P_est = P0_scaled.copy()

    for _ in range(steps):
        P_pred = P_est + Q_scaled
        H, r   = H_and_r_general(m_est)
        S      = H @ P_pred @ H.T + R_big
        K      = P_pred @ H.T @ inv(S)
        m_est  = m_est + K @ (-r)
        IKH    = np.eye(n_params) - K @ H
        P_est  = IKH @ P_pred @ IKH.T + K @ R_big @ K.T
        P_est  = 0.5 * (P_est + P_est.T)

    return m_est


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))))


def wrench_metrics(
    f_est: np.ndarray,
    l_est: np.ndarray,
    f_gt: np.ndarray,
    l_gt: np.ndarray,
) -> Dict[str, float]:
    force_err  = float(np.linalg.norm(f_est - f_gt))
    moment_err = float(np.linalg.norm(l_est - l_gt))
    fn = float(np.linalg.norm(f_gt))
    mn = float(np.linalg.norm(l_gt))
    return {
        "force_err_N":        force_err,
        "moment_err_Nm":      moment_err,
        "force_dir_err_deg":  _angle_deg(f_est, f_gt),
        "moment_dir_err_deg": _angle_deg(l_est, l_gt),
        "nrmse_force":        force_err  / (fn + 1e-12),
        "nrmse_moment":       moment_err / (mn + 1e-12),
        "force_gt_norm_N":    fn,
        "moment_gt_norm_Nm":  mn,
    }


# ---------------------------------------------------------------------------
# Per-order evaluation
# ---------------------------------------------------------------------------

def evaluate_one_order(
    positions_gt: np.ndarray,     # (N, M, 3)
    orientations_gt: np.ndarray,  # (N, M, 9)
    tau_gt: np.ndarray,           # (N, 4)
    f_ext_gt: np.ndarray,         # (N, 3)
    l_ext_gt: np.ndarray,         # (N, 3)
    case_ids_gt: np.ndarray,      # (N,)
    imu_datasets: Dict[str, Dict],
    order: ModalOrder,
    gamma: int,
    meas_std_deg: float,
    alpha: float,
    steps: int,
    L_phys: float,
) -> List[Dict]:
    """
    Evaluate oracle and EKF wrench estimation for one modal order.

    For oracle: uses GT backbone directly — no noise realization loop.
    For EKF: averages over noise realizations per layout.
    """
    results: List[Dict] = []
    num_cases = len(positions_gt)
    ox, oy, oz = order.order_x, order.order_y, order.order_z

    # --- Oracle: one result per case (GT backbone, no noise) ---
    print(f"  Oracle fitting {order.label} ...")
    t0 = time.perf_counter()
    for ci in range(num_cases):
        try:
            m_oracle = estimate_modal_oracle(
                positions_gt[ci], orientations_gt[ci], ox, oy, oz, L_phys)
            f_est, l_est = wrench_from_modal(m_oracle, tau_gt[ci], ox, oy, oz, gamma)
            metrics = wrench_metrics(f_est, l_est, f_ext_gt[ci], l_ext_gt[ci])
            valid = True
        except Exception as exc:
            metrics = wrench_metrics(np.zeros(3), np.zeros(3),
                                     f_ext_gt[ci], l_ext_gt[ci])
            valid = False

        row = {
            "case_id":       int(case_ids_gt[ci]),
            "method":        "oracle",
            "layout_name":   "GT",
            "modal_order":   order.label,
            "noise_real":    -1,
            "valid":         valid,
        }
        row.update(metrics)
        results.append(row)
    print(f"    Oracle done in {time.perf_counter()-t0:.1f} s")

    # --- EKF: one result per (case, noise_real, layout) ---
    for layout_name, ds in imu_datasets.items():
        R_meas_all   = ds["R_meas"]       # (N, n_noise, n_imu, 3, 3)
        imu_pos_norm = ds["imu_pos"]       # (n_imu,)
        num_noise    = R_meas_all.shape[1]

        print(f"  EKF {layout_name} {order.label} ({num_cases} cases × {num_noise} noise) ...")
        t0 = time.perf_counter()

        for ci in range(num_cases):
            for ni in range(num_noise):
                R_frame = [R_meas_all[ci, ni, si] for si in range(len(imu_pos_norm))]
                try:
                    if ox == 1 and oy == 1 and oz == 0:
                        # Standard 5-D EKF — use the validated run_ekf_on_frame
                        from evaluate_kirchhoff_shape_estimation import (
                            run_ekf_on_frame, P0_DIAG, STATE_DIM
                        )
                        import numpy as _np
                        m_ekf, _ = run_ekf_on_frame(
                            R_frame=R_frame,
                            imu_pos_norm=imu_pos_norm,
                            e3=E3.copy(),
                            gamma=gamma,
                            meas_std_deg=meas_std_deg,
                            alpha=alpha,
                            P0=np.diag(P0_DIAG),
                            steps=steps,
                        )
                    else:
                        m_ekf = run_ekf_for_order(
                            R_frame=R_frame,
                            imu_pos_norm=imu_pos_norm,
                            order_x=ox, order_y=oy, order_z=oz,
                            gamma=gamma,
                            meas_std_deg=meas_std_deg,
                            alpha=alpha,
                            steps=steps,
                        )
                    f_est, l_est = wrench_from_modal(m_ekf, tau_gt[ci], ox, oy, oz, gamma)
                    metrics = wrench_metrics(f_est, l_est, f_ext_gt[ci], l_ext_gt[ci])
                    valid = True
                except Exception:
                    metrics = wrench_metrics(np.zeros(3), np.zeros(3),
                                             f_ext_gt[ci], l_ext_gt[ci])
                    valid = False

                row = {
                    "case_id":       int(case_ids_gt[ci]),
                    "method":        "ekf",
                    "layout_name":   layout_name,
                    "modal_order":   order.label,
                    "noise_real":    ni,
                    "valid":         valid,
                }
                row.update(metrics)
                results.append(row)

        elapsed = time.perf_counter() - t0
        print(f"    EKF {layout_name} done in {elapsed:.1f} s")

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: List[Dict]) -> List[Dict]:
    from collections import defaultdict
    buckets: Dict[Tuple, List] = defaultdict(list)
    for r in results:
        key = (r["modal_order"], r["method"], r["layout_name"])
        buckets[key].append(r)

    summary: List[Dict] = []
    metric_keys = [
        "force_err_N", "moment_err_Nm",
        "force_dir_err_deg", "moment_dir_err_deg",
        "nrmse_force", "nrmse_moment",
    ]
    for (order_lbl, method, layout), rows in sorted(buckets.items()):
        entry: Dict = {
            "modal_order": order_lbl,
            "method":      method,
            "layout_name": layout,
            "n_runs":      len(rows),
        }
        for mk in metric_keys:
            vals = [r[mk] for r in rows if mk in r]
            if vals:
                entry[f"{mk}_mean"] = float(np.mean(vals))
                entry[f"{mk}_std"]  = float(np.std(vals))
                entry[f"{mk}_med"]  = float(np.median(vals))
        summary.append(entry)
    return summary


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV  -> {path}")


def save_json(results: List[Dict], summary: List[Dict], cfg: dict, path: Path) -> None:
    blob = {"config": cfg, "summary": summary, "results": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, default=str)
    print(f"  JSON -> {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _violin(ax, pos, data, color, label):
    if len(data) == 0:
        return
    vp = ax.violinplot(data, positions=[pos], widths=0.55,
                       showmedians=True, showextrema=False)
    for pc in vp["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.50)
        pc.set_edgecolor(color)
    vp["cmedians"].set_color(color)
    vp["cmedians"].set_linewidth(2.0)
    ax.scatter([pos], [np.mean(data)], marker="D", s=24, color=color, zorder=5)


def fig_violin_per_order(
    results: List[Dict],
    order_labels: List[str],
    save_stem: Path,
    metric_key: str,
    ylabel: str,
    scale: float,
    title: str,
) -> None:
    """Violin plot: for each order, side-by-side oracle vs EKF (2-IMU, 3-IMU)."""
    # Panels: one per order label
    n_orders = len(order_labels)
    fig, axes = plt.subplots(1, n_orders, figsize=(4 * n_orders, 5), sharey=True)
    if n_orders == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=TITLE_FS + 1, fontweight="bold")

    positions_map = {
        "oracle-GT":    1,
        "ekf-2-IMU":    2,
        "ekf-3-IMU":    3,
    }
    colors_map = {
        "oracle-GT":  "#2ca02c",
        "ekf-2-IMU":  "#1f77b4",
        "ekf-3-IMU":  "#d62728",
    }
    tick_labels = ["Oracle\n(GT)", "EKF\n(2-IMU)", "EKF\n(3-IMU)"]

    for ax, order_lbl in zip(axes, order_labels):
        for tag, pos in positions_map.items():
            if tag == "oracle-GT":
                method, layout = "oracle", "GT"
            else:
                method = "ekf"
                layout = tag.split("-", 1)[1]   # "2-IMU" or "3-IMU"
            data = np.array([r[metric_key] * scale for r in results
                             if r["modal_order"] == order_lbl
                             and r["method"] == method
                             and r["layout_name"] == layout])
            _violin(ax, pos, data, colors_map[tag], tag)
            if len(data):
                ax.text(pos, np.mean(data) * 1.04, f"{np.mean(data):.1f}",
                        ha="center", va="bottom",
                        fontsize=ANN_FS, color=colors_map[tag], fontweight="bold")

        ax.set_title(f"Order {order_lbl}", fontsize=TITLE_FS, fontweight="bold")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(tick_labels, fontsize=TICK_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(ylabel, fontsize=LABEL_FS)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = save_stem.parent / (save_stem.name + f"_{metric_key.split('_')[0]}." + ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


def fig_bar_summary(
    summary: List[Dict],
    order_labels: List[str],
    save_stem: Path,
) -> None:
    """Grouped bar chart: mean force/moment error per (order, method)."""
    combos = [
        ("oracle", "GT",    "#2ca02c", "Oracle (GT)"),
        ("ekf",    "2-IMU", "#1f77b4", "EKF (2-IMU)"),
        ("ekf",    "3-IMU", "#d62728", "EKF (3-IMU)"),
    ]
    n_orders  = len(order_labels)
    n_combos  = len(combos)
    x         = np.arange(n_orders)
    bar_w     = 0.22
    offsets   = np.linspace(-(n_combos - 1) / 2, (n_combos - 1) / 2, n_combos) * bar_w

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Oracle vs EKF: Mean Wrench Error by Modal Order",
                 fontsize=TITLE_FS + 1, fontweight="bold")

    for ax, mk, ylabel, scale, unit in [
        (axes[0], "force_err_N",   "Force error",   1e3, "mN"),
        (axes[1], "moment_err_Nm", "Moment error",  1e3, "mN·m"),
    ]:
        for (method, layout, color, lbl), off in zip(combos, offsets):
            means, stds = [], []
            for order_lbl in order_labels:
                hit = [s for s in summary
                        if s["modal_order"] == order_lbl
                        and s["method"] == method
                        and s["layout_name"] == layout]
                if hit:
                    means.append(hit[0].get(f"{mk}_mean", 0.0) * scale)
                    stds.append(hit[0].get(f"{mk}_std",   0.0) * scale)
                else:
                    means.append(0.0); stds.append(0.0)

            bars = ax.bar(x + off, means, bar_w, yerr=stds, capsize=3,
                          color=color, alpha=0.82, edgecolor="k",
                          linewidth=0.6, label=lbl)
            for bar, mean in zip(bars, means):
                if mean > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, mean * 1.03,
                            f"{mean:.1f}", ha="center", va="bottom",
                            fontsize=ANN_FS - 1, fontweight="bold", color=color)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Order\n{lb}" for lb in order_labels], fontsize=TICK_FS)
        ax.set_ylabel(f"{ylabel} [{unit}]", fontsize=LABEL_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    axes[1].legend(fontsize=LEGEND_FS, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = save_stem.parent / (save_stem.name + "_bars." + ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


def fig_nrmse_bars(
    summary: List[Dict],
    order_labels: List[str],
    save_stem: Path,
) -> None:
    """NRMSE bar chart to complement absolute error bars."""
    combos = [
        ("oracle", "GT",    "#2ca02c", "Oracle (GT)"),
        ("ekf",    "2-IMU", "#1f77b4", "EKF (2-IMU)"),
        ("ekf",    "3-IMU", "#d62728", "EKF (3-IMU)"),
    ]
    n_orders = len(order_labels)
    n_combos = len(combos)
    x        = np.arange(n_orders)
    bar_w    = 0.22
    offsets  = np.linspace(-(n_combos - 1) / 2, (n_combos - 1) / 2, n_combos) * bar_w

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Oracle vs EKF: NRMSE by Modal Order",
                 fontsize=TITLE_FS + 1, fontweight="bold")

    for ax, mk, ylabel in [
        (axes[0], "nrmse_force",  "Force NRMSE"),
        (axes[1], "nrmse_moment", "Moment NRMSE"),
    ]:
        for (method, layout, color, lbl), off in zip(combos, offsets):
            means = []
            for order_lbl in order_labels:
                hit = [s for s in summary
                        if s["modal_order"] == order_lbl
                        and s["method"] == method
                        and s["layout_name"] == layout]
                means.append(hit[0].get(f"{mk}_mean", 0.0) * 100 if hit else 0.0)

            bars = ax.bar(x + off, means, bar_w, color=color, alpha=0.82,
                          edgecolor="k", linewidth=0.6, label=lbl)
            for bar, mean in zip(bars, means):
                if mean > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, mean * 1.03,
                            f"{mean:.0f}%", ha="center", va="bottom",
                            fontsize=ANN_FS - 1, fontweight="bold", color=color)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Order\n{lb}" for lb in order_labels], fontsize=TICK_FS)
        ax.set_ylabel(f"{ylabel} [%]", fontsize=LABEL_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    axes[1].legend(fontsize=LEGEND_FS)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = save_stem.parent / (save_stem.name + "_nrmse." + ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oracle vs EKF wrench estimation comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt",       default="gt_data/kirchhoff_gt_dataset.npz")
    parser.add_argument("--imu2",     default="gt_data/kirchhoff_imu_2imu.npz")
    parser.add_argument("--imu3",     default="gt_data/kirchhoff_imu_3imu.npz")
    parser.add_argument("--save-dir", default="gt_data/results")
    parser.add_argument("--gamma",    type=int,   default=GAMMA_DEFAULT)
    parser.add_argument("--alpha",    type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--meas-std-deg", type=float, default=MEAS_STD_DEG_DEF)
    parser.add_argument("--steps",    type=int,   default=STEPS_DEFAULT)
    parser.add_argument(
        "--orders", nargs="+",
        choices=["(1,1,0)", "(1,2,0)", "(2,2,0)", "(2,2,1)"],
        default=["(1,1,0)", "(1,2,0)", "(2,2,0)", "(2,2,1)"],
        help="modal orders to evaluate",
    )
    parser.add_argument("--max-cases", type=int, default=None,
                        help="process only the first N cases after loading")
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("--max-cases must be > 0")

    script_dir = Path(__file__).resolve().parent

    def _resolve(p: str, must_exist: bool = True) -> Path:
        raw = Path(p)
        out = raw if raw.is_absolute() else (script_dir / raw).resolve()
        if must_exist and not out.exists():
            raise FileNotFoundError(out)
        return out

    gt_path   = _resolve(args.gt)
    imu2_path = _resolve(args.imu2, must_exist=False)
    imu3_path = _resolve(args.imu3, must_exist=False)
    save_dir  = _resolve(args.save_dir, must_exist=False)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_stem = save_dir / "oracle_vs_ekf"

    # --- Load GT ---
    print("Loading ground-truth dataset ...")
    positions_gt, orientations_gt, case_ids_gt, gt_meta = \
        load_ground_truth_dataset(gt_path)
    gt_data  = np.load(gt_path, allow_pickle=True)
    tau_gt   = gt_data["tau"]
    f_ext_gt = gt_data["f_ext"]
    l_ext_gt = gt_data["l_ext"]
    num_cases, num_pts, _ = positions_gt.shape
    if args.max_cases is not None:
        n_use = min(args.max_cases, num_cases)
        print(f"Applying --max-cases {args.max_cases}: using first {n_use} cases.")
        positions_gt = positions_gt[:n_use]
        orientations_gt = orientations_gt[:n_use]
        case_ids_gt = case_ids_gt[:n_use]
        tau_gt = tau_gt[:n_use]
        f_ext_gt = f_ext_gt[:n_use]
        l_ext_gt = l_ext_gt[:n_use]
        num_cases = n_use
    L_phys = float(gt_meta.get("length_m", _L))
    print(f"  {num_cases} cases, {num_pts} arc-length points, L = {L_phys} m")

    # --- Load IMU measurements ---
    imu_datasets: Dict[str, Dict] = {}
    for label, path in [("2-IMU", imu2_path), ("3-IMU", imu3_path)]:
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {label}")
            continue
        R_true, R_meas, imu_pos, imu_idx, imu_meta = load_imu_measurements(path)
        R_meas = R_meas[:num_cases]
        imu_actual_s = np.array(
            imu_meta.get("imu_actual_s", (imu_idx / (num_pts - 1)).tolist()),
            dtype=float,
        )
        imu_datasets[label] = {"R_meas": R_meas, "imu_pos": imu_actual_s}
        print(f"  {label}: R_meas {R_meas.shape},  imu_pos = "
              + "{" + ", ".join(f"{v:.4f}" for v in imu_actual_s) + "}")

    if not imu_datasets:
        raise RuntimeError("No IMU files found. Run Step 2 first.")

    # Filter modal orders to requested set
    orders_to_run = [o for o in MODAL_ORDERS if o.label in args.orders]

    # --- Evaluate each order ---
    all_results: List[Dict] = []
    for order in orders_to_run:
        print(f"\n{'='*60}")
        print(f"  Modal order {order.label}  "
              f"(nx={order.order_x}, ny={order.order_y}, nz={order.order_z})")
        print(f"{'='*60}")
        r = evaluate_one_order(
            positions_gt=positions_gt,
            orientations_gt=orientations_gt,
            tau_gt=tau_gt,
            f_ext_gt=f_ext_gt,
            l_ext_gt=l_ext_gt,
            case_ids_gt=case_ids_gt,
            imu_datasets=imu_datasets,
            order=order,
            gamma=args.gamma,
            meas_std_deg=args.meas_std_deg,
            alpha=args.alpha,
            steps=args.steps,
            L_phys=L_phys,
        )
        all_results.extend(r)

    # --- Aggregate and print ---
    summary = aggregate(all_results)
    order_labels = [o.label for o in orders_to_run]

    print(f"\n{'='*90}")
    print(f"  Oracle vs EKF Summary")
    print(f"{'='*90}")
    hdr = (f"{'Order':<12} {'Method':<8} {'Layout':<10} "
           f"{'Force [mN]':>14}  {'Moment [mN·m]':>16}  "
           f"{'NRMSE-F':>9}  {'NRMSE-M':>9}")
    print(hdr)
    print("-" * 90)
    for s in summary:
        ferr  = s.get("force_err_N_mean",    0.0) * 1e3
        fstd  = s.get("force_err_N_std",     0.0) * 1e3
        merr  = s.get("moment_err_Nm_mean",  0.0) * 1e3
        mstd  = s.get("moment_err_Nm_std",   0.0) * 1e3
        nrmf  = s.get("nrmse_force_mean",    0.0) * 100
        nrmm  = s.get("nrmse_moment_mean",   0.0) * 100
        print(f"{s['modal_order']:<12} {s['method']:<8} {s['layout_name']:<10} "
              f"{ferr:>8.2f}±{fstd:<5.2f}mN  "
              f"{merr:>9.3f}±{mstd:<5.3f}mN·m  "
              f"{nrmf:>7.1f}%  {nrmm:>7.1f}%")
    print("=" * 90)

    # --- Save ---
    print("\nSaving results ...")
    save_csv(all_results, save_dir / "oracle_vs_ekf_results.csv")
    save_csv(summary,     save_dir / "oracle_vs_ekf_summary.csv")
    save_json(
        all_results, summary,
        cfg={
            "gamma":        args.gamma,
            "alpha":        args.alpha,
            "meas_std_deg": args.meas_std_deg,
            "steps":        args.steps,
            "L_phys_m":     L_phys,
            "EIx_Nm2":      _EIX,
            "EIy_Nm2":      _EIY,
            "GJ_Nm2":       _GJ,
        },
        path=save_dir / "oracle_vs_ekf_results.json",
    )

    # --- Figures ---
    print("Generating figures ...")
    fig_violin_per_order(
        all_results, order_labels, save_stem,
        metric_key="force_err_N",
        ylabel="Force error [mN]",
        scale=1e3,
        title="Force Error: Oracle vs EKF by Modal Order",
    )
    fig_violin_per_order(
        all_results, order_labels, save_stem,
        metric_key="moment_err_Nm",
        ylabel="Moment error [mN·m]",
        scale=1e3,
        title="Moment Error: Oracle vs EKF by Modal Order",
    )
    fig_bar_summary(summary, order_labels, save_stem)
    fig_nrmse_bars(summary, order_labels, save_stem)
    print("Done.")


if __name__ == "__main__":
    main()
