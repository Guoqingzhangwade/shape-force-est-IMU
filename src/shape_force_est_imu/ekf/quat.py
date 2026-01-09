"""Quaternion helpers (xyzw convention, scalar last)."""

from __future__ import annotations

import numpy as np

from .common import skew


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for xyzw quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate of unit quaternion (inverse for unit quats)."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def q_fix_sign(q: np.ndarray) -> np.ndarray:
    """Keep scalar part positive to enforce a consistent sign."""
    return -q if q[3] < 0 else q


def quat_error(q_meas: np.ndarray, q_pred: np.ndarray) -> np.ndarray:
    """Minimal 3-vector orientation error for xyzw quaternions."""
    q_err = quat_mul(q_meas, quat_conj(q_pred))
    q_err = q_fix_sign(q_err)
    return 2.0 * q_err[:3]


def J_theta_q(q_meas: np.ndarray) -> np.ndarray:
    """Jacobian of theta wrt measured quaternion (3x4)."""
    qv, q0 = q_meas[:3], q_meas[3]
    return 2.0 * np.hstack([-qv.reshape(3, 1), -q0 * np.eye(3) - skew(qv)])


def J_q_q(q_meas: np.ndarray) -> np.ndarray:
    """Jacobian of q_err wrt q_meas (4x4)."""
    qv, q0 = q_meas[:3], q_meas[3]
    row0 = np.hstack([q0, qv])
    rows3 = np.hstack([qv.reshape(3, 1), -q0 * np.eye(3) - skew(qv)])
    return np.vstack([row0, rows3])


def J_q_R(Rm: np.ndarray) -> np.ndarray:
    """Jacobian of quaternion wrt rotation matrix (4x9)."""
    R11, R12, R13, R21, R22, R23, R31, R32, R33 = Rm.reshape(-1)
    q0 = 0.5 * np.sqrt(max(1.0 + R11 + R22 + R33, 1e-12))
    q = np.array(
        [
            (R32 - R23) / (4 * q0),
            (R13 - R31) / (4 * q0),
            (R21 - R12) / (4 * q0),
            q0,
        ]
    )
    J = np.zeros((4, 9))
    for idx in (0, 4, 8):
        J[3, idx] = 1.0 / (8 * q0)
    pos = {(0, 7), (1, 2), (2, 3)}
    neg = {(0, 5), (1, 6), (2, 1)}
    for n in range(3):
        for idx in range(9):
            if (n, idx) in pos:
                J[n, idx] = 1.0 / (4 * q0) - q[n] / (8 * q0**2)
            elif (n, idx) in neg:
                J[n, idx] = -1.0 / (4 * q0) - q[n] / (8 * q0**2)
            elif idx in (0, 4, 8):
                J[n, idx] = -q[n] / (8 * q0**2)
    return J
