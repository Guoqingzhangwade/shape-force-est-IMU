"""
Constrained tip-wrench solver — core module.

Solves the regularised constrained least-squares problem

    z_hat = argmin_z  || J_{Vbm}^T(m) S z - b_w ||_2^2
                    + lambda * || z - z0 ||_{Wz}^2

    w_hat = S z_hat          (body frame, [Mx,My,Mz,Fx,Fy,Fz]^T)

Wrench ordering follows virtual_work.vee() convention:
    rows 0:3 = moment_body,  rows 3:6 = force_body

Selection matrices (body / tip frame)
--------------------------------------
  S_FORCE_ONLY  (6×3)  zero body moments, 3-DOF body force
  S_DIR         (6×1)  single known direction d in body frame, scalar magnitude
  S_TRANSVERSE  (6×2)  zero body moments + zero body Fz, 2-DOF transverse force

Usage
-----
  from wrench_tip_constrained_solver import (
      make_S_force_only, make_S_direction, make_S_transverse,
      compute_b_w, solve_constrained_wrench,
      world_wrench_from_body, wrench_metrics,
  )
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

from virtual_work import (
    body_jacobian_at_s,
    cable_jacobian,
    elastic_energy_gradient,
    gram_matrix,
)
from scipy.linalg import block_diag as _block_diag

# ---------------------------------------------------------------------------
# Rod constants (must match the rest of the study)
# ---------------------------------------------------------------------------
_L        = 0.1
_E        = 60e9
_NU       = 0.3
_G        = _E / (2 * (1 + _NU))
_R_BB     = 5e-4
_I_BB     = np.pi * _R_BB**4 / 4.0
_EIX      = _E * _I_BB
_EIY      = _EIX
_GJ       = _G * 2 * _I_BB
_R_TENDON = 0.008
_R_LIST   = [
    np.array([ _R_TENDON,  0.0,        0.0]),
    np.array([ 0.0,         _R_TENDON, 0.0]),
    np.array([-_R_TENDON,  0.0,        0.0]),
    np.array([ 0.0,        -_R_TENDON, 0.0]),
]
_ORDER_X  = 1
_ORDER_Y  = 1
_ORDER_Z  = 0
_GAMMA    = 10   # product-of-exponentials segments

LAMBDA_DEFAULT = 1e-4   # regularisation strength (easy to tune here)

# ---------------------------------------------------------------------------
# Selection matrices
# ---------------------------------------------------------------------------

def make_S_force_only() -> np.ndarray:
    """6×3: body [Fx,Fy,Fz], zero body moments."""
    S = np.zeros((6, 3))
    S[3, 0] = S[4, 1] = S[5, 2] = 1.0
    return S


def make_S_direction(d_body: np.ndarray) -> np.ndarray:
    """6×1: wrench = [0,0,0, alpha*d]^T where d is a unit body-frame direction.

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
    """6×2: body [Fx,Fy] only (zero moments, zero Fz)."""
    S = np.zeros((6, 2))
    S[3, 0] = S[4, 1] = 1.0
    return S


# ---------------------------------------------------------------------------
# Generalised modal load
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
    b_w = grad_m U(m) - J_{lm}^T(m) tau
    """
    gradU = elastic_energy_gradient(m, EIx, EIy, GJ, L, order_x, order_y, order_z)
    J_lm  = cable_jacobian(m, r_list, L, order_x, order_y, order_z)
    return gradU - J_lm.T @ tau


# ---------------------------------------------------------------------------
# Regularised constrained solver
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
    Regularised constrained least-squares tip-wrench estimate.

    Problem
    -------
    w_b = S z  (constraint)
    z_hat = argmin_z || A z - b_w ||^2 + lam * || z - z0 ||_{Wz}^2
    A = J_{Vbm}^T(m) S   (n_params × n_z)

    Closed-form solution (ridge regression on reduced variable z):
    z_hat = (A^T A + lam Wz)^{-1} (A^T b_w + lam Wz z0)

    Parameters
    ----------
    m    : (n_params,) modal state
    tau  : (4,) cable tensions [N]
    S    : (6, n_z) selection matrix
    lam  : regularisation strength
    z0   : (n_z,) prior on z (default: zeros)
    Wz   : (n_z, n_z) regularisation weight matrix (default: identity)

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

    b_w          = compute_b_w(m, tau, order_x, order_y, order_z, L=L)
    J_vbm, T_tip = body_jacobian_at_s(m, 1.0, gamma, L, order_x, order_y, order_z)

    A     = J_vbm.T @ S                            # (n_params, n_z)
    lhs   = A.T @ A + lam * Wz                     # (n_z, n_z)
    rhs   = A.T @ b_w + lam * Wz @ z0             # (n_z,)
    z_hat = np.linalg.solve(lhs, rhs)              # (n_z,)
    w_body = S @ z_hat                             # (6,) [moment; force]

    return w_body, z_hat, J_vbm, T_tip


# ---------------------------------------------------------------------------
# Frame conversion
# ---------------------------------------------------------------------------

def world_wrench_from_body(
    w_body: np.ndarray,
    T_tip: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert body-frame wrench to world frame.

    w_body = [Mx,My,Mz, Fx,Fy,Fz]^T  (body frame)
    f_world = R_tip @ w_body[3:]
    l_world = R_tip @ w_body[:3]
    """
    R_tip   = T_tip[:3, :3]
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
    fe  = float(np.linalg.norm(f_est - f_gt))
    me  = float(np.linalg.norm(l_est - l_gt))
    fn  = float(np.linalg.norm(f_gt))
    mn  = float(np.linalg.norm(l_gt))
    return {
        "force_err_N"        : fe,
        "moment_err_Nm"      : me,
        "force_dir_err_deg"  : _angle_deg(f_est, f_gt),
        "moment_dir_err_deg" : _angle_deg(l_est, l_gt),
        "nrmse_force"        : fe / fn  if fn > 1e-10 else float("nan"),
        "nrmse_moment"       : me / mn  if mn > 1e-10 else float("nan"),
        "force_gt_norm_N"    : fn,
        "moment_gt_norm_Nm"  : mn,
    }
