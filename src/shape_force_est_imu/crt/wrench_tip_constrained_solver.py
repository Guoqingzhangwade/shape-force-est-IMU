"""
Constrained tip-wrench solver - core module.

Virtual-work convention
-----------------------
The tendon coordinate is pull/shortening q = ell_0 - ell. The tendon
pull/shortening Jacobian is

    J_qm = d q/dm = -d ell/dm

and the generalized external-wrench modal load is

    b_w = gradU - J_qm.T @ tau.

Wrench vectors are ordered

    [Mx, My, Mz, Fx, Fy, Fz] = [moment; force].

The direct full-6D pseudoinverse is a baseline. The practical constrained
estimator solves in a load subspace:

    F_b = S z
    J_Vbm.T S z = b_w.

Selection matrices (body / tip frame)
-------------------------------------
  S_FORCE_ONLY  (6x3)  zero body moments, 3-DOF body force
  S_DIR         (6x1)  single known direction d in body frame, scalar magnitude
  S_TRANSVERSE  (6x2)  zero body moments + zero body Fz, 2-DOF transverse force

Usage
-----
  from wrench_tip_constrained_solver import (
      make_S_force_only, make_S_direction, make_S_transverse,
      compute_b_w, solve_load_subspace_wrench, solve_constrained_wrench,
      world_wrench_from_body, wrench_metrics,
  )
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from virtual_work import (
    body_jacobian_at_s,
    elastic_energy_gradient,
    generalized_modal_load,
    pull_jacobian,
)

# ---------------------------------------------------------------------------
# Rod constants (must match the rest of the study)
# ---------------------------------------------------------------------------
_L = 0.1
_E = 60e9
_NU = 0.3
_G = _E / (2 * (1 + _NU))
_R_BB = 5e-4
_I_BB = np.pi * _R_BB**4 / 4.0
_EIX = _E * _I_BB
_EIY = _EIX
_GJ = _G * 2 * _I_BB
_R_TENDON = 0.008
_R_LIST = [
    np.array([_R_TENDON, 0.0, 0.0]),
    np.array([0.0, _R_TENDON, 0.0]),
    np.array([-_R_TENDON, 0.0, 0.0]),
    np.array([0.0, -_R_TENDON, 0.0]),
]
_ORDER_X = 1
_ORDER_Y = 1
_ORDER_Z = 0
_GAMMA = 10

LAMBDA_DEFAULT = 1e-4


# ---------------------------------------------------------------------------
# Selection matrices
# ---------------------------------------------------------------------------

def make_S_force_only() -> np.ndarray:
    """6x3: body [Fx,Fy,Fz], zero body moments."""
    S = np.zeros((6, 3))
    S[3, 0] = S[4, 1] = S[5, 2] = 1.0
    return S


def make_S_direction(d_body: np.ndarray) -> np.ndarray:
    """
    6x1: wrench = [0,0,0, alpha*d]^T where d is a unit body-frame direction.

    Parameters
    ----------
    d_body : (3,) unit vector in body (tip) frame
    """
    d = np.asarray(d_body, dtype=float)
    d = d / np.linalg.norm(d)
    S = np.zeros((6, 1))
    S[3:6, 0] = d
    return S


def make_S_transverse() -> np.ndarray:
    """6x2: body [Fx,Fy] only (zero moments, zero Fz)."""
    S = np.zeros((6, 2))
    S[3, 0] = S[4, 1] = 1.0
    return S


# ---------------------------------------------------------------------------
# Generalized modal load
# ---------------------------------------------------------------------------

def compute_b_w(
    m: np.ndarray,
    tau: np.ndarray,
    order_x: int = _ORDER_X,
    order_y: int = _ORDER_Y,
    order_z: int = _ORDER_Z,
    EIx: float = _EIX,
    EIy: float = _EIY,
    GJ: float = _GJ,
    L: float = _L,
    r_list=_R_LIST,
) -> np.ndarray:
    """
    b_w = grad_m U(m) - J_qm.T(m) tau
    """
    gradU = elastic_energy_gradient(m, EIx, EIy, GJ, L, order_x, order_y, order_z)
    J_qm = pull_jacobian(m, r_list, L, order_x, order_y, order_z)
    return generalized_modal_load(gradU, J_qm, tau)


def solve_reduced_map(
    A_S: np.ndarray,
    b_w: np.ndarray,
    Sigma_b: np.ndarray | None = None,
    z_prior: np.ndarray | None = None,
    Q_z: np.ndarray | None = None,
    rcond: float = 1e-10,
    jitter: float = 1e-12,
) -> np.ndarray:
    """
    Solve the residual-level reduced MAP problem:

        min_z ||A_S z - b_w||^2_{Sigma_b^{-1}}
            + ||z - z_prior||^2_{Q_z^{-1}}.

    If Sigma_b and Q_z are both None, this reduces to the minimum-norm
    least-squares solution pinv(A_S) @ b_w.
    """
    A_S = np.asarray(A_S, dtype=float)
    b_w = np.asarray(b_w, dtype=float)
    n_z = A_S.shape[1]

    if Sigma_b is None and Q_z is None:
        return np.linalg.pinv(A_S, rcond=rcond) @ b_w

    if z_prior is None:
        z_prior = np.zeros(n_z)
    else:
        z_prior = np.asarray(z_prior, dtype=float)

    W_b = (
        np.eye(A_S.shape[0])
        if Sigma_b is None
        else np.linalg.pinv(Sigma_b, rcond=rcond)
    )
    lhs = A_S.T @ W_b @ A_S
    rhs = A_S.T @ W_b @ b_w

    if Q_z is not None:
        Q_z_inv = np.linalg.pinv(Q_z, rcond=rcond)
        lhs = lhs + Q_z_inv
        rhs = rhs + Q_z_inv @ z_prior

    lhs = lhs + jitter * np.eye(n_z)
    return np.linalg.solve(lhs, rhs)


def solve_load_subspace_wrench(
    m: np.ndarray,
    tau: np.ndarray,
    S: np.ndarray,
    Sigma_b: np.ndarray | None = None,
    z_prior: np.ndarray | None = None,
    Q_z: np.ndarray | None = None,
    gamma: int = _GAMMA,
    order_x: int = _ORDER_X,
    order_y: int = _ORDER_Y,
    order_z: int = _ORDER_Z,
    L: float = _L,
    rcond: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Main constrained/MAP wrench estimator.

    Wrench model:
        F_b = S z

    Residual equation:
        J_Vbm.T S z = b_w

    Returns:
        w_body, z_hat, J_vbm, T_tip, b_w
    """
    b_w = compute_b_w(m, tau, order_x, order_y, order_z, L=L)
    J_vbm, T_tip = body_jacobian_at_s(m, 1.0, gamma, L, order_x, order_y, order_z)
    A_S = J_vbm.T @ S
    z_hat = solve_reduced_map(
        A_S,
        b_w,
        Sigma_b=Sigma_b,
        z_prior=z_prior,
        Q_z=Q_z,
        rcond=rcond,
    )
    w_body = S @ z_hat
    return w_body, z_hat, J_vbm, T_tip, b_w


# ---------------------------------------------------------------------------
# Regularized legacy constrained solver
# ---------------------------------------------------------------------------

def solve_constrained_wrench(
    m: np.ndarray,
    tau: np.ndarray,
    S: np.ndarray,
    lam: float = LAMBDA_DEFAULT,
    z0: np.ndarray | None = None,
    Wz: np.ndarray | None = None,
    gamma: int = _GAMMA,
    order_x: int = _ORDER_X,
    order_y: int = _ORDER_Y,
    order_z: int = _ORDER_Z,
    L: float = _L,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Legacy unweighted ridge-style constrained least-squares tip-wrench estimate.

    Problem
    -------
    w_b = S z  (constraint)
    z_hat = argmin_z || A z - b_w ||^2 + lam * || z - z0 ||_{Wz}^2
    A = J_Vbm.T(m) S   (n_params x n_z)

    Closed-form solution:
    z_hat = (A.T A + lam Wz)^-1 (A.T b_w + lam Wz z0)

    Returns
    -------
    w_body  : (6,) body-frame wrench [Mx,My,Mz,Fx,Fy,Fz]
    z_hat   : (n_z,) reduced parameter estimate
    J_vbm   : (6, n_params) body Jacobian (for diagnostics)
    T_tip   : (4, 4) tip homogeneous transform
    """
    n_z = S.shape[1]
    if z0 is None:
        z0 = np.zeros(n_z)
    if Wz is None:
        Wz = np.eye(n_z)

    b_w = compute_b_w(m, tau, order_x, order_y, order_z, L=L)
    J_vbm, T_tip = body_jacobian_at_s(m, 1.0, gamma, L, order_x, order_y, order_z)

    A = J_vbm.T @ S
    lhs = A.T @ A + lam * Wz
    rhs = A.T @ b_w + lam * Wz @ z0
    z_hat = np.linalg.solve(lhs, rhs)
    w_body = S @ z_hat

    return w_body, z_hat, J_vbm, T_tip


# ---------------------------------------------------------------------------
# Frame conversion
# ---------------------------------------------------------------------------

def world_wrench_from_body(
    w_body: np.ndarray,
    T_tip: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a body-frame wrench to a world-frame wrench about the same tip
    origin using rotation only.

    w_body = [Mx, My, Mz, Fx, Fy, Fz]^T  (body frame)
    moment_world_tip = R_tip @ moment_body
    force_world = R_tip @ force_body

    This function does not add p x f.
    """
    R_tip = T_tip[:3, :3]
    f_world = R_tip @ w_body[3:]
    l_world = R_tip @ w_body[:3]
    return f_world, l_world


# ---------------------------------------------------------------------------
# Metrics (world frame)
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
) -> dict:
    """All errors in world frame."""
    fe = float(np.linalg.norm(f_est - f_gt))
    me = float(np.linalg.norm(l_est - l_gt))
    fn = float(np.linalg.norm(f_gt))
    mn = float(np.linalg.norm(l_gt))
    return {
        "force_err_N": fe,
        "moment_err_Nm": me,
        "force_dir_err_deg": _angle_deg(f_est, f_gt),
        "moment_dir_err_deg": _angle_deg(l_est, l_gt),
        "nrmse_force": fe / fn if fn > 1e-10 else float("nan"),
        "nrmse_moment": me / mn if mn > 1e-10 else float("nan"),
        "force_gt_norm_N": fn,
        "moment_gt_norm_Nm": mn,
    }
