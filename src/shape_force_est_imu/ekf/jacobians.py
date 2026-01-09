"""Analytic Jacobians for curvature-model kinematics."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .common import skew
from .kinematics import curvature_kappa, magnus_4th_subinterval, twist_matrix


def d_kappa_dm(s: float, j: int, *, model: str = "5d") -> np.ndarray:
    v = np.zeros(3)
    if model == "5d":
        if j == 0:
            v[0] = 1.0
        elif j == 1:
            v[0] = s
        elif j == 2:
            v[1] = 1.0
        elif j == 3:
            v[1] = s
        else:
            v[2] = 1.0
        return v
    if model == "6d":
        if j == 0:
            v[0] = 1.0
        elif j == 1:
            v[0] = s
        elif j == 2:
            v[1] = 1.0
        elif j == 3:
            v[1] = s
        elif j == 4:
            v[2] = 1.0
        else:
            v[2] = s
        return v
    raise ValueError(f"Unsupported model: {model}")


def d_eta_dm(s: float, j: int, *, model: str = "5d", e3: np.ndarray | None = None) -> np.ndarray:
    k = d_kappa_dm(s, j, model=model)
    if e3 is None:
        e3 = np.array([0.0, 0.0, 1.0])
    return np.vstack([np.hstack([skew(k), np.zeros((3, 1))]), np.zeros((1, 4))])


def dPsi_dm(
    s0: float,
    s1: float,
    m: np.ndarray,
    j: int,
    *,
    model: str = "5d",
    e3: np.ndarray | None = None,
) -> np.ndarray:
    h = s1 - s0
    c1 = s0 + h * (0.5 - np.sqrt(3) / 6)
    c2 = s0 + h * (0.5 + np.sqrt(3) / 6)
    eta1 = twist_matrix(curvature_kappa(c1, m, model=model), e3)
    eta2 = twist_matrix(curvature_kappa(c2, m, model=model), e3)
    d_eta1 = d_eta_dm(c1, j, model=model, e3=e3)
    d_eta2 = d_eta_dm(c2, j, model=model, e3=e3)
    part1 = (h / 2) * (d_eta1 + d_eta2)
    part2 = (h**2 * np.sqrt(3) / 12) * (
        (d_eta1 @ eta2 + eta1 @ d_eta2) - (d_eta2 @ eta1 + eta2 @ d_eta1)
    )
    return part1 + part2


def dexp_first(A: np.ndarray, dA: np.ndarray) -> np.ndarray:
    """1st-order Bernoulli approximation."""
    return dA + 0.5 * (A @ dA - dA @ A)


def dexp_so3(A: np.ndarray, dA: np.ndarray) -> np.ndarray:
    """Exact right-Jacobian for so(3): returns J · dA."""
    a = np.array([A[2, 1], A[0, 2], A[1, 0]])
    theta = np.linalg.norm(a)
    if theta < 1e-9:
        return dA + 0.5 * (A @ dA - dA @ A)
    J = (
        (np.sin(theta) / theta) * np.eye(3)
        + (1 - np.sin(theta) / theta) * (a[:, None] @ a[None, :]) / (theta**2)
        + (1 - np.cos(theta)) / theta * A / theta
    )
    return J @ dA


def dexp_se3(A: np.ndarray, dA: np.ndarray) -> np.ndarray:
    """SE(3) lift: exact SO(3) for rotation, 1st-order for translation."""
    dR = dexp_so3(A[:3, :3], dA[:3, :3])
    dP = dA[:3, 3:] + 0.5 * (A[:3, :3] @ dA[:3, 3:] - dA[:3, :3] @ A[:3, 3:])
    out = np.zeros_like(dA)
    out[:3, :3] = dR
    out[:3, 3:] = dP
    return out


def dExp_dm(A: np.ndarray, dA: np.ndarray, *, mode: str = "exact") -> np.ndarray:
    """Derivative of expm(A) wrt A using exact (default) or approx mode."""
    if mode == "approx":
        return expm(A) @ dexp_first(A, dA)
    if mode == "exact":
        return expm(A) @ dexp_se3(A, dA)
    raise ValueError(f"Unsupported mode: {mode}")


def dT_dm(
    m: np.ndarray,
    s: float,
    j: int,
    *,
    gamma: int = 10,
    model: str = "5d",
    e3: np.ndarray | None = None,
    mode: str = "exact",
) -> np.ndarray:
    d = np.linspace(0.0, s, gamma + 1)
    Es, Ps = [], []
    for k in range(1, gamma + 1):
        P = magnus_4th_subinterval(d[k - 1], d[k], m, model=model, e3=e3)
        Es.append(expm(P))
        Ps.append(P)
    left = np.eye(4)
    dT = np.zeros((4, 4))
    for k in range(gamma):
        right = np.eye(4)
        for r in range(k + 1, gamma):
            right = right @ Es[r]
        dA = dPsi_dm(d[k], d[k + 1], m, j, model=model, e3=e3)
        dEk = dExp_dm(Ps[k], dA, mode=mode)
        dT += left @ dEk @ right
        left = left @ Es[k]
    return dT


def JRm_block(
    m: np.ndarray,
    s: float,
    *,
    gamma: int = 10,
    model: str = "5d",
    e3: np.ndarray | None = None,
    mode: str = "exact",
) -> np.ndarray:
    JRm = np.zeros((9, m.size))
    for j in range(m.size):
        JRm[:, j] = dT_dm(m, s, j, gamma=gamma, model=model, e3=e3, mode=mode)[
            :3, :3
        ].reshape(9)
    return JRm
