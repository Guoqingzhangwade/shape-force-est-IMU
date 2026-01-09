"""Curvature model and PoE/Magnus kinematics."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .common import skew


def phi(s: float, *, model: str = "5d") -> np.ndarray:
    """Modal basis Phi(s) for curvature model."""
    if model == "5d":
        return np.array([[1.0, s, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, s, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 1.0]])
    if model == "6d":
        return np.array([[1.0, s, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, s, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 1.0, s]])
    raise ValueError(f"Unsupported model: {model}")


def curvature_kappa(s: float, m: np.ndarray, *, model: str = "5d") -> np.ndarray:
    """Curvature vector kappa(s)."""
    return phi(s, model=model) @ m


def twist_matrix(kappa: np.ndarray, e3: np.ndarray | None = None) -> np.ndarray:
    """SE(3) twist matrix from curvature."""
    if e3 is None:
        e3 = np.array([0.0, 0.0, 1.0])
    top = np.hstack([skew(kappa), e3.reshape(3, 1)])
    return np.vstack([top, np.zeros((1, 4))])


def magnus_4th_subinterval(
    s0: float,
    s1: float,
    m: np.ndarray,
    *,
    model: str = "5d",
    e3: np.ndarray | None = None,
) -> np.ndarray:
    """4th-order Magnus term over subinterval [s0, s1]."""
    h = s1 - s0
    c1 = s0 + h * (0.5 - np.sqrt(3) / 6)
    c2 = s0 + h * (0.5 + np.sqrt(3) / 6)
    eta1 = twist_matrix(curvature_kappa(c1, m, model=model), e3)
    eta2 = twist_matrix(curvature_kappa(c2, m, model=model), e3)
    return (h / 2) * (eta1 + eta2) + (h**2 * np.sqrt(3) / 12) * (
        eta1 @ eta2 - eta2 @ eta1
    )


def product_of_exponentials(
    m: np.ndarray,
    s: float,
    *,
    gamma: int = 10,
    L: float = 100.0,
    model: str = "5d",
    e3: np.ndarray | None = None,
) -> np.ndarray:
    """Homogeneous transform at arclength s using Magnus PoE."""
    d = np.linspace(0.0, s, gamma + 1)
    T = np.eye(4)
    for k in range(1, gamma + 1):
        T = T @ expm(magnus_4th_subinterval(d[k - 1], d[k], m, model=model, e3=e3))
    T[:3, 3] *= L
    return T


def forward_kinematics_multiple(
    m: np.ndarray,
    s_vals,
    *,
    gamma: int = 10,
    L: float = 100.0,
    model: str = "5d",
    e3: np.ndarray | None = None,
):
    """Return lists of rotation matrices and positions at s_vals."""
    R_list, p_list = [], []
    for s in s_vals:
        T = product_of_exponentials(m, s, gamma=gamma, L=L, model=model, e3=e3)
        R_list.append(T[:3, :3])
        p_list.append(T[:3, 3])
    return R_list, p_list


def rotation_from_poE(
    m: np.ndarray,
    s: float,
    *,
    gamma: int = 10,
    L: float = 100.0,
    model: str = "5d",
    e3: np.ndarray | None = None,
) -> np.ndarray:
    """Rotation matrix at arclength s."""
    return product_of_exponentials(m, s, gamma=gamma, L=L, model=model, e3=e3)[:3, :3]
