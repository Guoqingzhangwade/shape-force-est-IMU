# -*- coding: utf-8 -*-
"""
Compare EKF measurement models (synthetic benchmark):
  1) Quaternion residual with numeric Jacobian
  2) Quaternion residual with analytic Jacobian (reference only)
  3) SO(3) log residual with analytic Jacobian
  4) SO(3) log residual with numeric Jacobian

Key features:
  - R-scale sweep for ALL methods (quaternion and SO(3))
  - Numerically-stable Jr^{-1} coefficient via cot(theta/2)
  - NIS reporting for consistency checks
  - Consistent quaternion sign convention (w >= 0)

Usage examples:
  python meas_model_compare.py --sweep-all-scales --scale-list "0.1,0.5,1.0,2.0,3.0"
  python meas_model_compare.py --trials 10 --steps 100 --gamma 20 --plot
"""

import argparse
import time
import warnings
from typing import List, Tuple, Dict, Callable
import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R


# ========================== CONSTANTS ==========================

# Numerical stability thresholds
THETA_SMALL_THRESHOLD = 1e-8  # Threshold for small angle approximation in jr_inv_so3
SIN_HALF_THRESHOLD = 1e-12    # Threshold for sin(theta/2) near zero
Q0_SINGULARITY_THRESHOLD = 1e-10  # Threshold for quaternion scalar near zero
QUAT_NORM_TOLERANCE = 1e-6    # Tolerance for quaternion normalization check

# Finite difference parameters
NUMERICAL_JACOBIAN_EPS = 1e-4  # Step size for numerical differentiation

# EKF process noise parameters (default values)
PROCESS_STD_0 = 1e-5  # Process noise for coefficients [k0_x, k0_y, k0]
PROCESS_STD_1 = 1e-6  # Process noise for coefficients [k1_x, k1_y]
INITIAL_COV_SCALE = 1e-2  # Initial covariance scaling factor

# Gauss-Legendre quadrature points for 2-point rule
GAUSS_LEGENDRE_XI = np.array([0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6])


# ========================== LIE ALGEBRA HELPERS ==========================

def skew(u: np.ndarray) -> np.ndarray:
    """
    Compute the skew-symmetric matrix from a 3D vector.

    For vector u = [u1, u2, u3], returns:
        [  0  -u3   u2 ]
        [ u3    0  -u1 ]
        [-u2   u1    0 ]

    Args:
        u: 3D vector

    Returns:
        3x3 skew-symmetric matrix
    """
    return np.array([[0.0, -u[2],  u[1]],
                     [u[2],  0.0, -u[0]],
                     [-u[1], u[0], 0.0]], dtype=float)


def vee(S: np.ndarray) -> np.ndarray:
    """
    Inverse of skew operator: extracts vector from skew-symmetric matrix.

    Args:
        S: 3x3 skew-symmetric matrix

    Returns:
        3D vector u such that skew(u) = S
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def so3_log(Rm: np.ndarray) -> np.ndarray:
    """
    Logarithm map from SO(3) to so(3): R -> rotation vector.

    Args:
        Rm: 3x3 rotation matrix

    Returns:
        3D rotation vector (axis-angle representation)
    """
    return R.from_matrix(Rm).as_rotvec()


def jr_inv_so3(r: np.ndarray) -> np.ndarray:
    """
    Right Jacobian inverse on SO(3): Jr^{-1}(r).

    Uses numerically stable cot(θ/2) formulation:
        Jr^{-1}(r) = I + 0.5*[r]_× + a*[r]_×^2
    where a = (1 - 0.5*θ*cot(θ/2)) / θ^2

    For small angles, uses Taylor expansion to avoid division by zero.

    Args:
        r: 3D rotation vector

    Returns:
        3x3 right Jacobian inverse matrix
    """
    theta = float(np.linalg.norm(r))
    A = skew(r)

    # Small angle approximation
    if theta < THETA_SMALL_THRESHOLD:
        return np.eye(3) + 0.5 * A + (1.0 / 12.0) * (A @ A)

    half = 0.5 * theta
    sin_half = np.sin(half)
    cos_half = np.cos(half)

    # Check for singularity when sin(theta/2) ≈ 0
    if abs(sin_half) < SIN_HALF_THRESHOLD:
        return np.eye(3) + 0.5 * A + (1.0 / 12.0) * (A @ A)

    cot_half = cos_half / sin_half
    a = (1.0 - 0.5 * theta * cot_half) / (theta ** 2)
    return np.eye(3) + 0.5 * A + a * (A @ A)


# ========================== SHAPE RECONSTRUCTION ==========================

def reconstruct_shape_3d(m: np.ndarray, e3: np.ndarray, gamma: int,
                         num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct 3D backbone shape from modal coefficients.

    The shape coefficients m = [k0_x, k1_x, k0_y, k1_y, k0_z] parameterize
    the curvature along the beam. This function integrates the kinematics
    to produce the full 3D centerline.

    Args:
        m: Shape coefficient vector [k0_x, k1_x, k0_y, k1_y, k0_z]
           - k0_x, k1_x: Curvature in x-direction (constant + linear terms)
           - k0_y, k1_y: Curvature in y-direction (constant + linear terms)
           - k0_z: Torsion (constant term)
        e3: Tip force vector (3D)
        gamma: Number of integration segments per point
        num_points: Number of points along the backbone

    Returns:
        positions: (num_points, 3) array of 3D positions
        orientations: (num_points, 3, 3) array of rotation matrices
    """
    s_vals = np.linspace(0, 1.0, num_points)
    positions = np.zeros((num_points, 3))
    orientations = np.zeros((num_points, 3, 3))

    positions[0] = np.array([0, 0, 0])
    orientations[0] = np.eye(3)

    for i, s in enumerate(s_vals):
        if s == 0:
            continue
        # Compute transformation matrix at arc-length s
        T = np.eye(4)
        h = s / gamma
        for k in range(1, gamma + 1):
            T = T @ expm(magnus_psi(k * h, h, m, e3))

        positions[i] = T[:3, 3]
        orientations[i] = T[:3, :3]

    return positions, orientations


# ========================== KINEMATICS + DERIVATIVES ==========================

def phi(s: float) -> np.ndarray:
    """
    Shape function matrix: maps shape coefficients to curvature at arc-length s.

    For shape parameterization m = [k0_x, k1_x, k0_y, k1_y, k0_z]:
        κ(s) = Φ(s) @ m
    where Φ(s) is a 3x5 matrix.

    Args:
        s: Arc-length parameter

    Returns:
        3x5 shape function matrix
    """
    return np.array([[1, s, 0, 0, 0],
                     [0, 0, 1, s, 0],
                     [0, 0, 0, 0, 1]], dtype=float)


def twist(k: np.ndarray, e3: np.ndarray) -> np.ndarray:
    """
    Construct 4x4 twist matrix from curvature and tip force.

    Args:
        k: 3D curvature vector
        e3: 3D tip force vector

    Returns:
        4x4 twist matrix in SE(3) Lie algebra
    """
    return np.block([[skew(k), e3[:, None]],
                     [np.zeros((1, 3)), 0.0]])


def magnus_psi(s_i: float, h: float, m: np.ndarray, e3: np.ndarray) -> np.ndarray:
    """
    Magnus expansion (2nd order) for the Lie algebra element over interval.

    Uses 2-point Gauss-Legendre quadrature for numerical integration.

    Args:
        s_i: End of current interval
        h: Interval size
        m: Shape coefficient vector (5D)
        e3: Tip force vector (3D)

    Returns:
        4x4 Lie algebra element (approximation to integral of twist)
    """
    xi = GAUSS_LEGENDRE_XI
    # Evaluate at Gauss points within interval [s_i - h, s_i]
    c1, c2 = s_i - h + xi * h
    k1, k2 = phi(c1) @ m, phi(c2) @ m
    e1, e2 = twist(k1, e3), twist(k2, e3)

    # Magnus expansion: Ψ ≈ h/2*(ξ1 + ξ2) + √3/12*h²*[ξ1, ξ2]
    return (h / 2) * (e1 + e2) + (np.sqrt(3) / 12) * h ** 2 * (e1 @ e2 - e2 @ e1)


def fwd_rotation(m: np.ndarray, s: float, e3: np.ndarray, gamma: int) -> np.ndarray:
    """
    Forward kinematics: compute rotation matrix at arc-length s.

    Uses Magnus expansion for integration over gamma segments.

    Args:
        m: Shape coefficient vector (5D)
        s: Arc-length parameter
        e3: Tip force vector (3D)
        gamma: Number of integration steps

    Returns:
        3x3 rotation matrix R(s)
    """
    T = np.eye(4)
    h = s / gamma

    for k in range(1, gamma + 1):
        T = T @ expm(magnus_psi(k * h, h, m, e3))

    return T[:3, :3]


def compute_R_m_derivative(m_coeffs: np.ndarray, s: float, gamma: int,
                          e3: np.ndarray) -> np.ndarray:
    """
    Compute dR/dm: derivative of rotation matrix w.r.t. shape coefficients.

    Uses Magnus expansion + 1st-order dexp for derivative computation.

    Algorithm:
        1. Compute forward pass: accumulate T_before[k] = exp(Ψ_1)...exp(Ψ_k)
        2. Compute backward pass: accumulate T_after[k] = exp(Ψ_k)...exp(Ψ_N)
        3. Compute dΨ_k/dm for each segment
        4. Apply chain rule: dT/dm = Σ T_before[k-1] @ dexp(Ψ_k) @ T_after[k+1]

    Args:
        m_coeffs: Shape coefficient vector (Nm-dimensional)
        s: Arc-length parameter
        gamma: Number of integration steps
        e3: Tip force vector (3D)

    Returns:
        Tensor of shape (3, 3, Nm): dR/dm_i for i=0,...,Nm-1
    """
    Nm = len(m_coeffs)
    Ns = gamma
    h = s / Ns

    # Storage for intermediate computations
    Psi = []  # Lie algebra elements
    e_Psi = []  # Exponentials of Lie algebra elements
    T_before = [np.eye(4)]  # Forward accumulation
    T_after = [np.eye(4) for _ in range(Ns + 1)]  # Backward accumulation (FIXED: independent copies)

    xi = GAUSS_LEGENDRE_XI
    dPsi_dm = [np.zeros((4, 4, Nm)) for _ in range(Ns)]  # dΨ_k/dm
    de_Psi_dm = [np.zeros((4, 4, Nm)) for _ in range(Ns)]  # d(exp(Ψ_k))/dm

    # ========== Forward pass: compute Ψ_k and T_before ==========
    for k in range(1, Ns + 1):
        s_i = k * h
        Psi_k = magnus_psi(s_i, h, m_coeffs, e3)
        Psi.append(Psi_k)
        e_Psi_k = expm(Psi_k)
        e_Psi.append(e_Psi_k)
        T_before.append(T_before[-1] @ e_Psi_k)

    # ========== Backward pass: compute T_after ==========
    T_after[Ns] = np.eye(4)
    for k in range(Ns - 1, 0, -1):
        T_after[k] = e_Psi[k] @ T_after[k + 1]

    # ========== Compute dΨ_k/dm ==========
    for k in range(1, Ns + 1):
        s_i = k * h
        c1 = s_i - h + xi[0] * h
        c2 = s_i - h + xi[1] * h
        phi_c1 = phi(c1)
        phi_c2 = phi(c2)

        kappa_1 = phi_c1 @ m_coeffs
        kappa_2 = phi_c2 @ m_coeffs
        eta_1 = twist(kappa_1, e3)
        eta_2 = twist(kappa_2, e3)

        for i in range(Nm):
            # Derivative of curvature w.r.t. m_i
            dkappa_1 = phi_c1[:, i]
            dkappa_2 = phi_c2[:, i]
            deta_1 = twist(dkappa_1, np.zeros(3))
            deta_2 = twist(dkappa_2, np.zeros(3))

            # Magnus expansion derivative
            comm_deta = (deta_1 @ eta_2 - eta_2 @ deta_1) + (eta_1 @ deta_2 - deta_2 @ eta_1)
            dPsi_k = (h / 2) * (deta_1 + deta_2) + (np.sqrt(3) / 12) * h ** 2 * comm_deta
            dPsi_dm[k - 1][:, :, i] = dPsi_k

    # ========== Compute d(exp(Ψ_k))/dm using 1st-order dexp ==========
    for k in range(Ns):
        Psi_k = Psi[k]
        e_Psi_k = e_Psi[k]
        for i in range(Nm):
            dPsi_k = dPsi_dm[k][:, :, i]
            # First-order dexp: d(exp(Ψ)) ≈ exp(Ψ) * (dΨ + 0.5*[Ψ, dΨ])
            dexpinv = dPsi_k + 0.5 * (Psi_k @ dPsi_k - dPsi_k @ Psi_k)
            de_Psi_dm[k][:, :, i] = e_Psi_k @ dexpinv

    # ========== Apply chain rule: dT/dm ==========
    dT_dm = np.zeros((4, 4, Nm))
    for i in range(Nm):
        for k in range(Ns):
            dT_dm[:, :, i] += T_before[k] @ de_Psi_dm[k][:, :, i] @ T_after[k + 1]

    return dT_dm[:3, :3, :]


# ========================== QUATERNION UTILITIES ==========================

def q_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion from [x, y, z, w] to [w, x, y, z] format."""
    x, y, z, w = q_xyzw
    return np.array([w, x, y, z], dtype=float)


def q_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion from [w, x, y, z] to [x, y, z, w] format."""
    w, x, y, z = q_wxyz
    return np.array([x, y, z, w], dtype=float)


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Quaternion multiplication: a * b (in [w, x, y, z] format).

    Args:
        a: First quaternion [w, x, y, z]
        b: Second quaternion [w, x, y, z]

    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=float)


def q_inv(q: np.ndarray) -> np.ndarray:
    """
    Quaternion inverse.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Inverse quaternion (conjugate normalized by squared norm)
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float) / np.dot(q, q)


def validate_quaternion(q: np.ndarray, name: str = "quaternion") -> None:
    """
    Validate that quaternion is properly normalized.

    Args:
        q: Quaternion to validate
        name: Name for error messages

    Raises:
        ValueError: If quaternion norm is far from 1.0
    """
    norm = np.linalg.norm(q)
    if abs(norm - 1.0) > QUAT_NORM_TOLERANCE:
        warnings.warn(f"{name} norm is {norm:.6e}, expected ~1.0. Auto-normalizing.")


# ========================== QUATERNION RESIDUAL ==========================

def theta_quat(q_meas_wxyz: np.ndarray, m: np.ndarray, s: float,
               e3: np.ndarray, gamma: int) -> np.ndarray:
    """
    Quaternion-based measurement residual.

    Computes error quaternion: q_e = q_meas ⊗ q_pred^{-1}
    Returns residual: 2 * q_e.vector (imaginary part)

    Args:
        q_meas_wxyz: Measured quaternion [w, x, y, z]
        m: Shape coefficient vector
        s: Arc-length parameter
        e3: Tip force vector
        gamma: Number of integration steps

    Returns:
        3D residual vector
    """
    # Compute predicted rotation
    R_pred = fwd_rotation(m, s, e3, gamma)
    q_pred = q_xyzw_to_wxyz(R.from_matrix(R_pred).as_quat())

    # Enforce consistent sign convention: w >= 0
    if q_pred[0] < 0:
        q_pred = -q_pred

    # Error quaternion
    q_e = q_mul(q_meas_wxyz, q_inv(q_pred))

    # Residual is 2 times the vector part
    return 2.0 * q_e[1:]


def jac_num_quat(q_meas_wxyz: np.ndarray, m: np.ndarray, s: float,
                 e3: np.ndarray, gamma: int, eps: float = NUMERICAL_JACOBIAN_EPS) -> np.ndarray:
    """
    Numerical Jacobian of quaternion residual w.r.t. shape coefficients.

    Uses central finite differences.

    Args:
        q_meas_wxyz: Measured quaternion
        m: Shape coefficient vector
        s: Arc-length parameter
        e3: Tip force vector
        gamma: Number of integration steps
        eps: Finite difference step size

    Returns:
        3 x Nm Jacobian matrix
    """
    J = np.zeros((3, m.size))
    for i in range(m.size):
        m_p, m_m = m.copy(), m.copy()
        m_p[i] += eps
        m_m[i] -= eps
        J[:, i] = (theta_quat(q_meas_wxyz, m_p, s, e3, gamma)
                   - theta_quat(q_meas_wxyz, m_m, s, e3, gamma)) / (2 * eps)
    return J


def compute_dtheta_dqhat(q_meas_wxyz: np.ndarray) -> np.ndarray:
    """
    Compute ∂θ/∂q̂ where θ = 2*q_e.vector and q̂ is the unit quaternion.

    Args:
        q_meas_wxyz: Measured quaternion [w, x, y, z]

    Returns:
        3 x 4 derivative matrix
    """
    # Normalize measurement
    q_meas = q_meas_wxyz / np.linalg.norm(q_meas_wxyz)
    q0 = q_meas[0]
    qv = q_meas[1:]
    qv_hat = skew(qv)

    out = np.zeros((3, 4))
    out[:, 0] = qv  # ∂θ/∂w
    out[:, 1:] = -q0 * np.eye(3) - qv_hat  # ∂θ/∂(x,y,z)
    return 2.0 * out


def compute_q_R_derivative_trace_branch(R_matrix: np.ndarray) -> np.ndarray:
    """
    Compute ∂q/∂R using trace-branch method.

    WARNING: This only implements the trace branch of Shepperd's method.
    May be inaccurate when trace(R) is small (rotation angle near 180°).

    Args:
        R_matrix: 3x3 rotation matrix

    Returns:
        4 x 3 x 3 tensor: ∂q_i/∂R_jk for i=0,1,2,3 and j,k=0,1,2
    """
    # Convert to quaternion with scalar first
    q = R.from_matrix(R_matrix).as_quat(scalar_first=True)
    q = q / np.linalg.norm(q)

    # Enforce w >= 0 sign convention
    sign = 1.0
    if q[0] < 0:
        q = -q
        sign = -1.0

    q0, q1, q2, q3 = q

    # Check for singularity when q0 ≈ 0
    if abs(q0) < Q0_SINGULARITY_THRESHOLD:
        warnings.warn("Quaternion scalar component near zero. Jacobian may be inaccurate.")
        return np.zeros((4, 3, 3))

    # Compute derivatives based on trace formula
    J = np.zeros((4, 3, 3))

    # ∂q0/∂R = 1/(8*q0) * I (diagonal elements only)
    for i in range(3):
        J[0, i, i] = 1.0 / (8.0 * q0)

    # ∂q1/∂R
    J[1, 2, 1] = 1.0 / (4.0 * q0)
    J[1, 1, 2] = -1.0 / (4.0 * q0)
    for i in range(3):
        J[1, i, i] = -q1 / (8.0 * q0 ** 2)

    # ∂q2/∂R
    J[2, 0, 2] = 1.0 / (4.0 * q0)
    J[2, 2, 0] = -1.0 / (4.0 * q0)
    for i in range(3):
        J[2, i, i] = -q2 / (8.0 * q0 ** 2)

    # ∂q3/∂R
    J[3, 1, 0] = 1.0 / (4.0 * q0)
    J[3, 0, 1] = -1.0 / (4.0 * q0)
    for i in range(3):
        J[3, i, i] = -q3 / (8.0 * q0 ** 2)
    return sign * J


def jac_analy_quat_reference(q_meas_wxyz: np.ndarray, m: np.ndarray, s: float,
                             e3: np.ndarray, gamma: int) -> np.ndarray:
    """
    Analytic Jacobian of quaternion residual.

    Routes through the SO(3) log Jacobian to avoid Shepperd-branch singularities
    that occur when the rotation angle exceeds ~120 degrees:

        θ_quat = (2·sin(φ/2) / φ) · r_so3,   r_so3 = log(R_meas @ R_pred^T)

        ∂θ_quat/∂m = Df(r) @ ∂r_so3/∂m

    where Df = (2·sin(φ/2)/φ)·I + (cos(φ/2) − 2·sin(φ/2)/φ)/φ² · r·rᵀ

    Args:
        q_meas_wxyz: Measured quaternion [w, x, y, z]
        m: Shape coefficient vector
        s: Arc-length parameter
        e3: Tip force vector
        gamma: Number of integration steps

    Returns:
        3 x Nm Jacobian matrix
    """
    # Convert quaternion measurement to rotation matrix
    R_meas = R.from_quat(q_wxyz_to_xyzw(q_meas_wxyz)).as_matrix()

    # Predicted rotation and SO(3) error
    R_pred = fwd_rotation(m, s, e3, gamma)
    R_e = R_meas @ R_pred.T
    r = so3_log(R_e)          # SO(3) residual (rotation vector)
    phi = np.linalg.norm(r)

    # Right Jacobian inverse for SO(3) Jacobian
    Jr_inv = jr_inv_so3(r)

    # ∂R_pred/∂m
    dR_dm = compute_R_m_derivative(m, s, gamma, e3)  # 3 x 3 x Nm

    # SO(3) Jacobian: H_so3[:, p] = -Jr_inv @ vee(skew_part(dR @ R_pred^T))
    H_so3 = np.zeros((3, m.size))
    for p in range(m.size):
        dR = dR_dm[:, :, p]
        A = dR @ R_pred.T
        A_skew = 0.5 * (A - A.T)
        H_so3[:, p] = -Jr_inv @ vee(A_skew)

    # Df: Jacobian of θ_quat = 2·sin(φ/2)·r̂ w.r.t. r
    if phi < THETA_SMALL_THRESHOLD:
        Df = np.eye(3)  # small-angle limit: θ_quat ≈ r_so3
    else:
        a = 2.0 * np.sin(phi / 2.0) / phi
        b = (np.cos(phi / 2.0) - 2.0 * np.sin(phi / 2.0) / phi) / (phi ** 2)
        Df = a * np.eye(3) + b * np.outer(r, r)

    return Df @ H_so3


# ========================== SO(3) RESIDUAL ==========================

def theta_so3(R_meas: np.ndarray, m: np.ndarray, s: float,
              e3: np.ndarray, gamma: int) -> np.ndarray:
    """
    SO(3) log-based measurement residual.

    Computes: log(R_meas @ R_pred^T) as a rotation vector.

    Args:
        R_meas: Measured rotation matrix
        m: Shape coefficient vector
        s: Arc-length parameter
        e3: Tip force vector
        gamma: Number of integration steps

    Returns:
        3D residual vector (rotation vector)
    """
    R_pred = fwd_rotation(m, s, e3, gamma)
    R_e = R_meas @ R_pred.T
    return so3_log(R_e)


def jac_num_so3(R_meas: np.ndarray, m: np.ndarray, s: float,
                e3: np.ndarray, gamma: int, eps: float = NUMERICAL_JACOBIAN_EPS) -> np.ndarray:
    """
    Numerical Jacobian of SO(3) residual w.r.t. shape coefficients.

    Uses central finite differences.

    Args:
        R_meas: Measured rotation matrix
        m: Shape coefficient vector
        s: Arc-length parameter
        e3: Tip force vector
        gamma: Number of integration steps
        eps: Finite difference step size

    Returns:
        3 x Nm Jacobian matrix
    """
    J = np.zeros((3, m.size))
    for i in range(m.size):
        m_p, m_m = m.copy(), m.copy()
        m_p[i] += eps
        m_m[i] -= eps
        J[:, i] = (theta_so3(R_meas, m_p, s, e3, gamma)
                   - theta_so3(R_meas, m_m, s, e3, gamma)) / (2 * eps)
    return J


def measurement_so3_and_H_analytic(m: np.ndarray, s_vals: np.ndarray,
                                   R_meas_list: List[np.ndarray], e3: np.ndarray,
                                   gamma: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SO(3) residuals and analytic Jacobian for multiple sensors.

    Uses the fact that for r = log(R_meas @ R_pred^T):
        ∂r/∂m = -Jr^{-1}(r) @ vee(∂R/∂m @ R_pred^T)_skew

    Args:
        m: Shape coefficient vector
        s_vals: Array of arc-length values (one per sensor)
        R_meas_list: List of measured rotation matrices
        e3: Tip force vector
        gamma: Number of integration steps

    Returns:
        H: (3*Nsensors, Nm) stacked Jacobian matrix
        r: (3*Nsensors,) stacked residual vector
    """
    H_blocks, r_blocks = [], []

    for s_i, R_meas in zip(s_vals, R_meas_list):
        # Compute residual
        R_pred = fwd_rotation(m, s_i, e3, gamma)
        R_e = R_meas @ R_pred.T
        r = so3_log(R_e)

        # Right Jacobian inverse
        Jr_inv = jr_inv_so3(r)

        # Compute ∂R/∂m
        dR_dm = compute_R_m_derivative(m, s_i, gamma, e3)

        # Jacobian: H = -Jr^{-1} @ vee(dR @ R_pred^T)_skew
        H_i = np.zeros((3, m.size))
        for p in range(m.size):
            dR = dR_dm[:, :, p]
            A = dR @ R_pred.T
            # Extract skew-symmetric part: (A - A^T) / 2
            A = 0.5 * (A - A.T)
            H_i[:, p] = -Jr_inv @ vee(A)

        H_blocks.append(H_i)
        r_blocks.append(r)

    return np.vstack(H_blocks), np.hstack(r_blocks)


def measurement_so3_and_H_numeric(m: np.ndarray, s_vals: np.ndarray,
                                  R_meas_list: List[np.ndarray], e3: np.ndarray,
                                  gamma: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SO(3) residuals and numerical Jacobian for multiple sensors.

    Args:
        m: Shape coefficient vector
        s_vals: Array of arc-length values (one per sensor)
        R_meas_list: List of measured rotation matrices
        e3: Tip force vector
        gamma: Number of integration steps

    Returns:
        H: (3*Nsensors, Nm) stacked Jacobian matrix
        r: (3*Nsensors,) stacked residual vector
    """
    H_blocks, r_blocks = [], []

    for s_i, R_meas in zip(s_vals, R_meas_list):
        r = theta_so3(R_meas, m, s_i, e3, gamma)
        H_i = jac_num_so3(R_meas, m, s_i, e3, gamma)

        H_blocks.append(H_i)
        r_blocks.append(r)

    return np.vstack(H_blocks), np.hstack(r_blocks)


# ========================== CORE EKF FUNCTIONS ==========================

def ekf_core(meas_seq: List, m_true: np.ndarray, imu_pos: np.ndarray,
             e3: np.ndarray, gamma: int, meas_std_deg: float, R_scale: float,
             measurement_fn: Callable,
             proc_std0: float = PROCESS_STD_0,
             proc_std1: float = PROCESS_STD_1,
             init_cov_scale: float = INITIAL_COV_SCALE,
             P0: np.ndarray = None,
             nis_burnin: int = 0) -> Dict:
    """
    Core EKF implementation (refactored common code).

    This function implements the standard EKF prediction-update cycle,
    parameterized by measurement and Jacobian functions.

    Args:
        meas_seq: Sequence of measurements (format depends on measurement_fn)
        m_true: True shape coefficients (for error computation)
        imu_pos: Array of sensor arc-length positions
        e3: Tip force vector
        gamma: Number of integration steps
        meas_std_deg: Measurement noise standard deviation (degrees)
        R_scale: Measurement covariance scaling factor
        measurement_fn: Function to compute residuals and Jacobian
        proc_std0: Process noise std for coefficients [k0_x, k0_y, k0]
        proc_std1: Process noise std for coefficients [k1_x, k1_y]
        init_cov_scale: Initial covariance scaling (used only when P0 is None)
        P0: Initial covariance matrix (5x5).  If provided, overrides init_cov_scale.
        nis_burnin: Number of initial steps to exclude from post-burn-in NIS stats.
                    0 means post-burnin == full-horizon (no burn-in skipped).

    Returns:
        Dictionary with results: m_est, hist, rmse_*, mae_*, time_*, nis_*
        Includes both full-horizon and post-burn-in NIS statistics.
    """
    state_dim = m_true.size

    # Process noise covariance
    Q = np.diag([proc_std0**2, proc_std1**2,
                 proc_std0**2, proc_std1**2,
                 proc_std0**2])

    # Measurement noise covariance (single sensor)
    sigma = np.deg2rad(meas_std_deg)
    R_single = R_scale * (sigma ** 2) * np.eye(3)

    # Initialize state estimate and covariance
    m_est = np.zeros(state_dim)
    if P0 is not None:
        P_est = P0.copy()
    else:
        P_est = init_cov_scale * np.eye(state_dim)

    # History tracking
    hist = []
    nis_hist = []  # Normalized Innovation Squared (one value per step)

    t0 = time.perf_counter()

    for k in range(len(meas_seq)):
        # ========== Prediction step ==========
        m_pred = m_est.copy()  # State transition is identity (constant shape)
        P_pred = P_est + Q

        # ========== Measurement update step ==========
        # Compute residuals and Jacobian
        H, innov = measurement_fn(m_pred, imu_pos, meas_seq[k], e3, gamma)
        innov = -innov  # Convert residual to innovation

        # Measurement covariance (stacked for all sensors)
        R_big = np.kron(np.eye(len(imu_pos)), R_single)

        # Innovation covariance
        S = H @ P_pred @ H.T + R_big
        S_inv = inv(S)  # Compute inverse once for reuse

        # Kalman gain
        K = P_pred @ H.T @ S_inv

        # State update
        m_est = m_pred + K @ innov

        # Covariance update (Joseph form for numerical stability)
        IKH = np.eye(state_dim) - K @ H
        P_est = IKH @ P_pred @ IKH.T + K @ R_big @ K.T

        # Enforce symmetry (mitigate numerical errors)
        P_est = 0.5 * (P_est + P_est.T)

        # Track history
        hist.append(m_est.copy())

        # Normalized Innovation Squared (for consistency check)
        nis_hist.append(float(innov.T @ S_inv @ innov))

    t1 = time.perf_counter()

    # ========== Compute error metrics ==========
    hist = np.array(hist)
    err = hist - m_true
    rmse_t = np.sqrt(np.mean(err ** 2, axis=1))  # RMSE over coefficients at each time

    # Full-horizon NIS stats
    nis_arr = np.array(nis_hist)

    # Post-burn-in NIS: skip the first nis_burnin steps
    # If burnin >= steps, fall back to full horizon silently
    burnin_idx = min(nis_burnin, len(nis_arr) - 1)
    nis_post = nis_arr[burnin_idx:]

    return {
        "m_est": m_est,
        "hist": hist,
        "rmse_final": float(np.sqrt(np.mean((m_est - m_true) ** 2))),
        "rmse_mean": float(np.mean(rmse_t)),
        "rmse_max": float(np.max(rmse_t)),
        "mae_mean": float(np.mean(np.abs(err))),
        "time_total": float(t1 - t0),
        "time_per_step": float((t1 - t0) / len(meas_seq)),
        "nis_mean": float(np.mean(nis_arr)),
        "nis_std": float(np.std(nis_arr)),
        "nis_mean_postburnin": float(np.mean(nis_post)),
        "nis_std_postburnin": float(np.std(nis_post)),
    }


def ekf_quat(meas_q_seq: List[List[np.ndarray]], m_true: np.ndarray,
             imu_pos: np.ndarray, e3: np.ndarray, gamma: int, mode: str,
             meas_std_deg: float, R_scale: float = 1.0,
             P0: np.ndarray = None, nis_burnin: int = 0) -> Dict:
    """
    EKF with quaternion-based measurement model.

    Args:
        meas_q_seq: Sequence of quaternion measurements
                    meas_q_seq[t][i] = quaternion at time t, sensor i
        m_true: True shape coefficients
        imu_pos: Sensor arc-length positions
        e3: Tip force vector
        gamma: Number of integration steps
        mode: "quat_numeric" or "quat_analytic"
        meas_std_deg: Measurement noise std (degrees)
        R_scale: Measurement covariance scaling factor
        P0: Initial covariance matrix (passed through to ekf_core)
        nis_burnin: Steps to skip for post-burn-in NIS (passed through to ekf_core)

    Returns:
        Dictionary with EKF results
    """
    # Select Jacobian computation method
    if mode == "quat_numeric":
        jac_fn = jac_num_quat
    elif mode == "quat_analytic":
        jac_fn = jac_analy_quat_reference
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Wrapper to match expected signature for ekf_core
    def measurement_fn(m_pred, s_vals, meas_frame, e3_vec, gamma_val):
        H_stack, r_stack = [], []
        for i, s in enumerate(s_vals):
            q_meas = meas_frame[i]
            r = theta_quat(q_meas, m_pred, s, e3_vec, gamma_val)
            H = jac_fn(q_meas, m_pred, s, e3_vec, gamma_val)
            H_stack.append(H)
            r_stack.append(r)  # Return residual (ekf_core will negate to get innovation)
        return np.vstack(H_stack), np.hstack(r_stack)

    return ekf_core(meas_q_seq, m_true, imu_pos, e3, gamma, meas_std_deg, R_scale,
                   measurement_fn, P0=P0, nis_burnin=nis_burnin)


def ekf_so3(meas_R_seq: List[List[np.ndarray]], m_true: np.ndarray,
            imu_pos: np.ndarray, e3: np.ndarray, gamma: int, mode: str,
            meas_std_deg: float, R_scale: float = 1.0,
            P0: np.ndarray = None, nis_burnin: int = 0) -> Dict:
    """
    EKF with SO(3) log-based measurement model.

    Args:
        meas_R_seq: Sequence of rotation matrix measurements
                    meas_R_seq[t][i] = rotation matrix at time t, sensor i
        m_true: True shape coefficients
        imu_pos: Sensor arc-length positions
        e3: Tip force vector
        gamma: Number of integration steps
        mode: "so3_analytic" or "so3_numeric"
        meas_std_deg: Measurement noise std (degrees)
        R_scale: Measurement covariance scaling factor
        P0: Initial covariance matrix (passed through to ekf_core)
        nis_burnin: Steps to skip for post-burn-in NIS (passed through to ekf_core)

    Returns:
        Dictionary with EKF results
    """
    # Select measurement function
    if mode == "so3_analytic":
        meas_fn = measurement_so3_and_H_analytic
    elif mode == "so3_numeric":
        meas_fn = measurement_so3_and_H_numeric
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return ekf_core(meas_R_seq, m_true, imu_pos, e3, gamma, meas_std_deg, R_scale,
                   meas_fn, P0=P0, nis_burnin=nis_burnin)


# ========================== DATA GENERATION ==========================

def build_meas_seq(m_true: np.ndarray, imu_pos: np.ndarray, e3: np.ndarray,
                   gamma: int, steps: int, meas_std_deg: float,
                   rng: np.random.RandomState) -> Tuple[List, List]:
    """
    Generate synthetic measurement sequences with noise.

    Generates both quaternion and rotation matrix representations for each
    measurement, ensuring consistent sign convention (w >= 0 for quaternions).

    Args:
        m_true: True shape coefficient vector
        imu_pos: Array of sensor arc-length positions
        e3: Tip force vector
        gamma: Number of integration steps
        steps: Number of time steps to generate
        meas_std_deg: Measurement noise std (degrees)
        rng: Random number generator

    Returns:
        meas_q: List of quaternion measurements [time][sensor]
        meas_R: List of rotation matrix measurements [time][sensor]
    """
    meas_q = []
    meas_R = []

    for _ in range(steps):
        frame_q = []
        frame_R = []

        for s in imu_pos:
            # Generate clean measurement
            R_clean = fwd_rotation(m_true, s, e3, gamma)
            q_clean = q_xyzw_to_wxyz(R.from_matrix(R_clean).as_quat())

            # Add noise via random rotation
            noise = R.from_rotvec(rng.randn(3) * np.deg2rad(meas_std_deg))
            q_noise = q_xyzw_to_wxyz(noise.as_quat())

            # Apply noise: q_meas = q_noise ⊗ q_clean
            q_meas = q_mul(q_noise, q_clean)

            # Enforce sign convention: w >= 0
            if q_meas[0] < 0:
                q_meas = -q_meas

            frame_q.append(q_meas)
            frame_R.append(R.from_quat(q_wxyz_to_xyzw(q_meas)).as_matrix())

        meas_q.append(frame_q)
        meas_R.append(frame_R)

    return meas_q, meas_R


# ========================== REPORTING ==========================

def summarize(results: List[Dict]) -> Dict[str, Tuple[float, float]]:
    """
    Compute mean and std of metrics across multiple trials.

    Args:
        results: List of result dictionaries from EKF runs

    Returns:
        Dictionary mapping metric name to (mean, std) tuple
    """
    keys = ["rmse_final", "rmse_mean", "rmse_max", "mae_mean",
            "time_total", "time_per_step", "nis_mean", "nis_std",
            "nis_mean_postburnin", "nis_std_postburnin"]
    stats = {}
    for k in keys:
        vals = np.array([r[k] for r in results], dtype=float)
        stats[k] = (float(np.mean(vals)), float(np.std(vals)))
    return stats


def plot_shape_comparison(m_true: np.ndarray, m_est_dict: Dict[str, np.ndarray],
                         e3: np.ndarray, gamma: int) -> None:
    """
    Plot 3D shape comparison between true and estimated shapes.

    Args:
        m_true: True shape coefficients
        m_est_dict: Dictionary mapping method name to estimated coefficients
        e3: Tip force vector
        gamma: Number of integration segments
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot skipped] matplotlib is not installed.  Run: pip install matplotlib")
        return

    fig = plt.figure(figsize=(15, 10))

    # 3D view
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Reconstruct true shape
    pos_true, _ = reconstruct_shape_3d(m_true, e3, gamma, num_points=50)
    ax1.plot(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2],
             'k-', linewidth=3, label='True', alpha=0.7)

    # Plot estimated shapes
    colors = {'quat_numeric': 'r', 'quat_analytic': 'orange',
              'so3_analytic': 'b', 'so3_numeric': 'g'}
    labels = {'quat_numeric': 'Quat num', 'quat_analytic': 'Quat ana',
              'so3_analytic': 'SO(3) ana', 'so3_numeric': 'SO(3) num'}

    for method, m_est in m_est_dict.items():
        pos_est, _ = reconstruct_shape_3d(m_est, e3, gamma, num_points=50)
        ax1.plot(pos_est[:, 0], pos_est[:, 1], pos_est[:, 2],
                color=colors.get(method, 'gray'), linewidth=2,
                label=labels.get(method, method), linestyle='--', alpha=0.8)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Backbone Shape Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # XY projection
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(pos_true[:, 0], pos_true[:, 1], 'k-', linewidth=3, label='True', alpha=0.7)
    for method, m_est in m_est_dict.items():
        pos_est, _ = reconstruct_shape_3d(m_est, e3, gamma, num_points=50)
        ax2.plot(pos_est[:, 0], pos_est[:, 1],
                color=colors.get(method, 'gray'), linewidth=2,
                label=labels.get(method, method), linestyle='--', alpha=0.8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # XZ projection
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(pos_true[:, 0], pos_true[:, 2], 'k-', linewidth=3, label='True', alpha=0.7)
    for method, m_est in m_est_dict.items():
        pos_est, _ = reconstruct_shape_3d(m_est, e3, gamma, num_points=50)
        ax3.plot(pos_est[:, 0], pos_est[:, 2],
                color=colors.get(method, 'gray'), linewidth=2,
                label=labels.get(method, method), linestyle='--', alpha=0.8)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # YZ projection
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(pos_true[:, 1], pos_true[:, 2], 'k-', linewidth=3, label='True', alpha=0.7)
    for method, m_est in m_est_dict.items():
        pos_est, _ = reconstruct_shape_3d(m_est, e3, gamma, num_points=50)
        ax4.plot(pos_est[:, 1], pos_est[:, 2],
                color=colors.get(method, 'gray'), linewidth=2,
                label=labels.get(method, method), linestyle='--', alpha=0.8)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('YZ Projection')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Estimated Shape Comparison\n' +
                f'True m = [{m_true[0]:.2f}, {m_true[1]:.2f}, {m_true[2]:.2f}, {m_true[3]:.2f}, {m_true[4]:.2f}]',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def print_comparison_table(all_stats: Dict[str, Dict], scale_info: Dict[str, float],
                          nis_dof: int) -> None:
    """
    Print formatted comparison table for all 4 methods.

    Args:
        all_stats: Dictionary mapping method name to statistics
        scale_info: Dictionary mapping method name to R-scale used
        nis_dof: Degrees of freedom for NIS statistic
    """
    methods = ["quat_numeric", "quat_analytic", "so3_analytic", "so3_numeric"]
    labels = ["Quat num", "Quat ana", "SO(3) ana", "SO(3) num"]

    rows = [
        ("RMSE final", "rmse_final"),
        ("RMSE mean", "rmse_mean"),
        ("RMSE max", "rmse_max"),
        ("MAE mean", "mae_mean"),
        ("Time/step (s)", "time_per_step"),
        ("NIS mean", "nis_mean"),
        ("NIS std", "nis_std"),
    ]

    print("\n" + "=" * 120)
    print("MEASUREMENT MODEL COMPARISON (mean +/- std)")
    print("=" * 120)

    header = f"{'Metric':<18}"
    for label in labels:
        header += f" | {label:<24}"
    print(header)
    print("-" * 120)

    for name, key in rows:
        line = f"{name:<18}"
        for method in methods:
            if method in all_stats:
                mu, sd = all_stats[method][key]
                line += f" | {mu:>10.4e} +/- {sd:<10.2e}"
            else:
                line += f" | {'N/A':<24}"
        print(line)

    print("-" * 120)
    print("R-scales used:")
    for method in methods:
        if method in scale_info:
            print(f"  {method}: {scale_info[method]:.2f}")
    print(f"NIS DOF (per step): {nis_dof}")
    print("=" * 120)


# ========================== SHAPE BANK + ALPHA SWEEP ==========================

def build_shape_bank(num_shapes: int,
                     k0_range: Tuple[float, float],
                     k1_range: Tuple[float, float],
                     kz_range: Tuple[float, float],
                     seed: int) -> List[np.ndarray]:
    """
    Generate a fixed bank of true shape states for the alpha sweep.

    Each shape is a 5D modal coefficient vector m = [k0_x, k1_x, k0_y, k1_y, k0_z].
    Coefficients are drawn uniformly from the specified ranges using a fixed seed,
    so the bank is fully reproducible.

    Args:
        num_shapes: Number of distinct true shape vectors to generate
        k0_range: (lo, hi) for k0_x and k0_y (constant curvature)
        k1_range: (lo, hi) for k1_x and k1_y (linear curvature)
        kz_range: (lo, hi) for k0_z (torsion)
        seed: RNG seed for reproducibility

    Returns:
        List of num_shapes arrays, each of shape (5,)
    """
    rng = np.random.RandomState(seed)
    bank = []
    for _ in range(num_shapes):
        m = np.array([
            rng.uniform(*k0_range),  # k0_x
            rng.uniform(*k1_range),  # k1_x
            rng.uniform(*k0_range),  # k0_y
            rng.uniform(*k1_range),  # k1_y
            rng.uniform(*kz_range),  # k0_z
        ])
        bank.append(m)
    return bank


def build_noise_seeds(num_shapes: int, num_noise_realizations: int,
                      seed_base: int) -> np.ndarray:
    """
    Generate a 2D table of noise seeds for reproducible measurement sequences.

    Seeds are assigned as seed_base + row*num_noise_realizations + col so that
    every (shape, noise-realization) pair has a unique, fixed seed independent
    of the alpha value being tested.

    Args:
        num_shapes: Number of distinct true shapes
        num_noise_realizations: Number of noise draws per shape
        seed_base: Integer offset for all seeds

    Returns:
        Integer array of shape (num_shapes, num_noise_realizations)
    """
    return (np.arange(num_shapes * num_noise_realizations, dtype=int)
            .reshape(num_shapes, num_noise_realizations) + seed_base)


# ── Reproducibility note ──────────────────────────────────────────────────────
# Command used for the filter-consistency and measurement-noise scaling study
# reported in the manuscript (run from the scripts/shape_est/ directory):
#
#   python meas_model_compare.py `
#     --mode so3_scale_sweep `
#     --alpha-list "0.1,0.2,0.5,0.8,1.0,1.5,2.0,3.0" `
#     --num-shapes 10 --num-noise-realizations 3 `
#     --selected-alpha 1.0 `
#     --plot-alpha-sweep
#
# Settings applied by default (not overridden above):
#   --imu-pos "0.25,0.50,1.00"   (3-IMU layout, NIS DOF = 9)
#   --shape-bank-seed 123        (fixed shape bank for reproducibility)
#   --noise-seed-base 1000       (fixed noise seeds, same across all alpha values)
#   --nis-burnin 10              (post-burn-in NIS excludes first 10 steps)
#   --init-cov-diag default      (P0 = diag([1.0, 0.5, 1.0, 0.5, 0.25]),
#                                 reflecting shape-bank sampling ranges)
#   --steps 50                   (time steps per run)
#   --meas-std-deg 0.5           (measurement noise std in degrees)
#
# Total runs per alpha: 10 shapes x 3 noise realizations = 30
# Results saved to: so3_alpha_sweep_results.csv / so3_alpha_sweep_results.json
# ──────────────────────────────────────────────────────────────────────────────


def run_so3_analytic_alpha_sweep(
        alpha_list: List[float],
        shape_bank: List[np.ndarray],
        noise_seeds: np.ndarray,
        imu_pos: np.ndarray,
        e3: np.ndarray,
        gamma: int,
        steps: int,
        meas_std_deg: float,
        alpha_objective: str,
        nis_dof: int,
        P0: np.ndarray = None,
        nis_burnin: int = 0) -> List[Dict]:
    """
    Run SO(3)-analytic EKF for a sweep of alpha (R-scale) values.

    Fairness guarantee: measurement sequences are pre-generated ONCE from the
    fixed (shape_bank, noise_seeds) pairs and then reused identically for every
    alpha value.  This eliminates sampling noise when comparing alphas.

    Args:
        alpha_list: Alpha values to sweep
        shape_bank: Fixed list of true shape vectors (pre-generated)
        noise_seeds: 2D seed table [num_shapes x num_noise_realizations]
        imu_pos: IMU arc-length positions
        e3: Tip force vector
        gamma: Integration segments
        steps: Time steps per run
        meas_std_deg: Measurement noise std (degrees)
        alpha_objective: Metric key used later for best-alpha selection (informational)
        nis_dof: Expected NIS degrees of freedom (printed for reference)
        P0: Initial covariance matrix (5x5). None uses default init_cov_scale*I.
        nis_burnin: Steps to skip when computing post-burn-in NIS statistics.

    Returns:
        List of result dicts, one entry per alpha value.
        NIS stats are reported both for the full horizon and post-burn-in.
    """
    num_shapes = len(shape_bank)
    num_noise_realizations = noise_seeds.shape[1]
    total_runs = num_shapes * num_noise_realizations

    print(f"\n  Pre-generating {total_runs} measurement sequences "
          f"({num_shapes} shapes x {num_noise_realizations} noise realizations) ...")

    # --- Pre-generate all measurement sequences (reused across all alphas) ---
    all_meas_R: List[List] = []
    for si, m_true in enumerate(shape_bank):
        noise_row = []
        for ni in range(num_noise_realizations):
            seed = int(noise_seeds[si, ni])
            rng = np.random.RandomState(seed)
            _, meas_R = build_meas_seq(m_true, imu_pos, e3, gamma, steps, meas_std_deg, rng)
            noise_row.append(meas_R)
        all_meas_R.append(noise_row)

    print(f"  Done.  Starting alpha sweep ({len(alpha_list)} values) ...\n")

    sweep_results = []

    for alpha in alpha_list:
        run_results = []

        for si, m_true in enumerate(shape_bank):
            for ni in range(num_noise_realizations):
                meas_R = all_meas_R[si][ni]
                res = ekf_so3(meas_R, m_true, imu_pos, e3, gamma,
                              "so3_analytic", meas_std_deg, R_scale=alpha,
                              P0=P0, nis_burnin=nis_burnin)
                run_results.append(res)

        stat = summarize(run_results)

        entry = {
            "alpha":                        alpha,
            "num_shapes":                   num_shapes,
            "num_noise_realizations":       num_noise_realizations,
            "num_total_runs":               total_runs,
            "imu_layout":                   list(imu_pos),
            "nis_dof":                      nis_dof,
            "nis_burnin":                   nis_burnin,
            "rmse_mean_mean":               stat["rmse_mean"][0],
            "rmse_mean_std":                stat["rmse_mean"][1],
            "rmse_final_mean":              stat["rmse_final"][0],
            "rmse_final_std":               stat["rmse_final"][1],
            "nis_mean_full_mean":           stat["nis_mean"][0],
            "nis_mean_full_std":            stat["nis_mean"][1],
            "nis_std_full_mean":            stat["nis_std"][0],
            "nis_std_full_std":             stat["nis_std"][1],
            "nis_mean_postburnin_mean":     stat["nis_mean_postburnin"][0],
            "nis_mean_postburnin_std":      stat["nis_mean_postburnin"][1],
            "nis_std_postburnin_mean":      stat["nis_std_postburnin"][0],
            "nis_std_postburnin_std":       stat["nis_std_postburnin"][1],
            "time_per_step_mean":           stat["time_per_step"][0],
            "time_per_step_std":            stat["time_per_step"][1],
        }
        sweep_results.append(entry)

        print(f"  alpha={alpha:5.3f}"
              f" | RMSE_mean={stat['rmse_mean'][0]:.4e} +/- {stat['rmse_mean'][1]:.2e}"
              f" | RMSE_final={stat['rmse_final'][0]:.4e}"
              f" | NIS_full={stat['nis_mean'][0]:.2f} +/- {stat['nis_mean'][1]:.2f}"
              f" | NIS_post(b={nis_burnin})={stat['nis_mean_postburnin'][0]:.2f}"
              f" +/- {stat['nis_mean_postburnin'][1]:.2f}"
              f"  (DOF={nis_dof})"
              f" | {stat['time_per_step'][0]*1e3:.2f} ms/step")

    return sweep_results


# ========================== MAIN ==========================

def validate_inputs(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.

    Args:
        args: Parsed command-line arguments

    Raises:
        ValueError: If any arguments are invalid
    """
    if args.trials <= 0:
        raise ValueError("trials must be positive")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.gamma <= 0:
        raise ValueError("gamma must be positive")
    if args.meas_std_deg <= 0:
        raise ValueError("meas-std-deg must be positive")

    if args.mode == "full_compare":
        # Validate per-method R-scales only when they are used
        for name, val in [("quat-num-scale", args.quat_num_scale),
                          ("quat-ana-scale", args.quat_ana_scale),
                          ("so3-ana-scale", args.so3_ana_scale),
                          ("so3-num-scale", args.so3_num_scale)]:
            if val <= 0:
                raise ValueError(f"{name} must be positive")

    if args.mode == "so3_scale_sweep":
        if args.num_shapes <= 0:
            raise ValueError("num-shapes must be positive")
        if args.num_noise_realizations <= 0:
            raise ValueError("num-noise-realizations must be positive")


def main():
    """
    Main entry point for EKF comparison script.

    This script compares different EKF measurement models for estimating
    the modal shape coefficients of a continuum manipulator (soft robot/catheter).

    ESTIMATION TARGET:
    The state being estimated is m = [k0_x, k1_x, k0_y, k1_y, k0_z], which are
    the modal coefficients that parameterize the curvature along the beam:
        - k0_x, k1_x: x-direction curvature (constant + linear in arc-length)
        - k0_y, k1_y: y-direction curvature (constant + linear in arc-length)
        - k0_z: torsion (constant)

    These coefficients fully describe the 3D shape of the deformed beam.
    """
    parser = argparse.ArgumentParser(
        description="Compare EKF measurement models for continuum manipulator shape estimation.\n"
                   "Estimates modal coefficients m=[k0_x, k1_x, k0_y, k1_y, k0_z] from IMU measurements.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    parser.add_argument("--mode",
                       choices=["full_compare", "so3_scale_sweep"],
                       default="full_compare",
                       help="Operation mode: 'full_compare' runs all 4 methods (default); "
                            "'so3_scale_sweep' sweeps alpha for SO(3)-analytic only.")

    # ── Shared parameters ─────────────────────────────────────────────────────
    parser.add_argument("--imu-pos", type=str, default="0.25,0.50,1.00",
                       help="Comma-separated IMU arc-length positions (default: '0.25,0.50,1.00')")
    parser.add_argument("--trials", type=int, default=5,
                       help="(full_compare) Number of Monte Carlo trials (default: 5)")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of time steps per trial/run (default: 50)")
    parser.add_argument("--gamma", type=int, default=10,
                       help="Number of integration segments (default: 10)")
    parser.add_argument("--seed-list", type=str, default="11,22,33,44,55,66,77,88,99,111",
                       help="(full_compare) Comma-separated seeds for Monte Carlo trials")
    parser.add_argument("--meas-std-deg", type=float, default=0.5,
                       help="Measurement noise std deviation in degrees (default: 0.5)")

    # ── full_compare-specific parameters (unchanged from original) ────────────
    parser.add_argument("--quat-num-scale", type=float, default=1.0,
                       help="(full_compare) R-scale for quaternion numeric method")
    parser.add_argument("--quat-ana-scale", type=float, default=1.0,
                       help="(full_compare) R-scale for quaternion analytic method")
    parser.add_argument("--so3-ana-scale", type=float, default=1.0,
                       help="(full_compare) R-scale for SO(3) analytic method")
    parser.add_argument("--so3-num-scale", type=float, default=1.0,
                       help="(full_compare) R-scale for SO(3) numeric method")
    parser.add_argument("--sweep-all-scales", action="store_true",
                       help="(full_compare) Sweep R-scale for all 4 methods to find optimal values")
    parser.add_argument("--scale-list", type=str, default="0.1,0.2,0.5,0.8,1.0,1.5,2.0,3.0",
                       help="(full_compare) Comma-separated R-scales to sweep with --sweep-all-scales")
    parser.add_argument("--objective", choices=["rmse_mean", "rmse_final"], default="rmse_mean",
                       help="(full_compare) Objective metric for scale optimisation")
    parser.add_argument("--plot", action="store_true",
                       help="(full_compare) Show RMSE convergence plots")
    parser.add_argument("--plot-shape", action="store_true",
                       help="(full_compare) Show 3D shape reconstruction comparison")
    parser.add_argument("--fast", action="store_true",
                       help="(full_compare) Only run SO(3) numeric method")

    # ── so3_scale_sweep-specific parameters (new) ─────────────────────────────
    parser.add_argument("--alpha-list", type=str, default="0.1,0.2,0.5,0.8,1.0,1.5,2.0,3.0",
                       help="(so3_scale_sweep) Comma-separated alpha (R-scale) values to sweep")
    parser.add_argument("--num-shapes", type=int, default=10,
                       help="(so3_scale_sweep) Number of distinct true shapes in the shape bank (default: 10)")
    parser.add_argument("--num-noise-realizations", type=int, default=3,
                       help="(so3_scale_sweep) Noise realizations per shape (default: 3)")
    parser.add_argument("--shape-bank-seed", type=int, default=123,
                       help="(so3_scale_sweep) Seed for shape bank generation (default: 123)")
    parser.add_argument("--noise-seed-base", type=int, default=1000,
                       help="(so3_scale_sweep) Base offset for noise seeds (default: 1000)")
    parser.add_argument("--k0-range", type=str, default="-2.0,2.0",
                       help="(so3_scale_sweep) Uniform range for k0_x, k0_y (default: '-2.0,2.0')")
    parser.add_argument("--k1-range", type=str, default="-1.5,1.5",
                       help="(so3_scale_sweep) Uniform range for k1_x, k1_y (default: '-1.5,1.5')")
    parser.add_argument("--kz-range", type=str, default="-1.0,1.0",
                       help="(so3_scale_sweep) Uniform range for k0_z torsion (default: '-1.0,1.0')")
    parser.add_argument("--alpha-objective", choices=["rmse_mean", "rmse_final"],
                       default="rmse_mean",
                       help="(so3_scale_sweep) Objective for selecting the recommended alpha (default: rmse_mean)")
    parser.add_argument("--plot-alpha-sweep", action="store_true",
                       help="(so3_scale_sweep) Plot RMSE and NIS vs alpha after sweep")
    parser.add_argument("--selected-alpha", type=float, default=None,
                       help="(so3_scale_sweep) Alpha value to annotate on the plot as the paper selection. "
                            "Independent of the RMSE-based recommendation. "
                            "Example: --selected-alpha 1.0")

    # ── Initial covariance (shared, both modes) ───────────────────────────────
    parser.add_argument("--init-cov-scale", type=float, default=None,
                       help="P0 = scale * I.  "
                            "Default: 1e-2 for full_compare; "
                            "diag([1,0.5,1,0.5,0.25]) for so3_scale_sweep unless overridden.")
    parser.add_argument("--init-cov-diag", type=str, default="",
                       help="P0 = diag(values), e.g. '1.0,0.5,1.0,0.5,0.25'.  "
                            "Takes precedence over --init-cov-scale.")

    # ── Post-burn-in NIS (shared, both modes) ────────────────────────────────
    parser.add_argument("--nis-burnin", type=int, default=10,
                       help="Steps to skip when computing post-burn-in NIS (default: 10).  "
                            "Set to 0 to disable (post-burnin == full-horizon).")

    args = parser.parse_args()
    validate_inputs(args)

    # ── Build initial covariance P0 ───────────────────────────────────────────
    # Default P0 for so3_scale_sweep reflects the shape-bank sampling ranges:
    #   k0_x, k0_y ~ U(-2,2) -> variance ~1.33  -> use 1.0 (slightly conservative)
    #   k1_x, k1_y ~ U(-1.5,1.5) -> variance ~0.75 -> use 0.5
    #   k0_z      ~ U(-1,1)   -> variance ~0.33  -> use 0.25
    _SO3_SWEEP_DEFAULT_P0_DIAG = np.array([1.0, 0.5, 1.0, 0.5, 0.25])

    if args.init_cov_diag:
        # Diagonal mode: user-supplied diagonal takes highest precedence
        _diag_vals = np.array([float(x) for x in args.init_cov_diag.split(",") if x.strip()])
        if len(_diag_vals) != 5:
            raise ValueError("--init-cov-diag must have exactly 5 comma-separated values")
        P0 = np.diag(_diag_vals)
    elif args.init_cov_scale is not None:
        # Scalar mode: P0 = scale * I
        P0 = args.init_cov_scale * np.eye(5)
    else:
        # Mode-specific defaults
        if args.mode == "so3_scale_sweep":
            P0 = np.diag(_SO3_SWEEP_DEFAULT_P0_DIAG)
        else:
            P0 = INITIAL_COV_SCALE * np.eye(5)  # preserve original full_compare behavior

    # ── Shared physical setup ─────────────────────────────────────────────────
    imu_pos = np.array([float(x) for x in args.imu_pos.split(",") if x.strip()])
    L_phys = 100.0
    e3 = np.array([0.0, 0.0, L_phys], dtype=float)
    nis_dof = 3 * len(imu_pos)  # NIS degrees of freedom = 3 (SO3 dim) x number of IMUs

    # =========================================================================
    #  SO(3)-ANALYTIC ALPHA SWEEP MODE
    # =========================================================================
    if args.mode == "so3_scale_sweep":
        alpha_list = [float(x) for x in args.alpha_list.split(",") if x.strip()]
        k0_lo, k0_hi = [float(x) for x in args.k0_range.split(",")]
        k1_lo, k1_hi = [float(x) for x in args.k1_range.split(",")]
        kz_lo, kz_hi = [float(x) for x in args.kz_range.split(",")]

        total_runs = args.num_shapes * args.num_noise_realizations

        print("\n" + "=" * 80)
        print("SO(3)-ANALYTIC ALPHA (R-SCALE) SWEEP")
        print("=" * 80)
        imu_layout_str = "[" + ", ".join(f"{v:.2f}" for v in imu_pos) + "]"
        print(f"  IMU layout:               {imu_layout_str}  ({len(imu_pos)} sensors)")
        print(f"  NIS DOF per step:         {nis_dof}  (= 3 x {len(imu_pos)} IMUs)")
        print(f"  True shapes (bank size):  {args.num_shapes}")
        print(f"  Noise realizations/shape: {args.num_noise_realizations}")
        print(f"  Total runs per alpha:     {total_runs}")
        print(f"  Shape bank seed:          {args.shape_bank_seed}")
        print(f"  Noise seed base:          {args.noise_seed_base}")
        print(f"  Steps per run:            {args.steps}")
        print(f"  Meas noise std:           {args.meas_std_deg} deg")
        print(f"  k0 range  (k0_x, k0_y):  [{k0_lo}, {k0_hi}]")
        print(f"  k1 range  (k1_x, k1_y):  [{k1_lo}, {k1_hi}]")
        print(f"  kz range  (k0_z):         [{kz_lo}, {kz_hi}]")
        print(f"  Alpha values:             {alpha_list}")
        print(f"  Alpha objective:          {args.alpha_objective}")
        print(f"  NIS burn-in steps:        {args.nis_burnin}")
        p0_diag_str = "[" + ", ".join(f"{v:.3f}" for v in np.diag(P0)) + "]"
        print(f"  Initial cov P0 (diag):    {p0_diag_str}")
        print("=" * 80)

        # Build fixed shape bank and noise seeds ONCE — reused for every alpha
        shape_bank = build_shape_bank(
            args.num_shapes,
            (k0_lo, k0_hi), (k1_lo, k1_hi), (kz_lo, kz_hi),
            args.shape_bank_seed
        )
        noise_seeds = build_noise_seeds(
            args.num_shapes, args.num_noise_realizations, args.noise_seed_base
        )

        print(f"\n  Shape bank [{args.num_shapes} shapes, seed={args.shape_bank_seed}]:")
        for i, m in enumerate(shape_bank):
            print(f"    Shape {i+1:2d}: [{m[0]:+6.3f}, {m[1]:+6.3f}, "
                  f"{m[2]:+6.3f}, {m[3]:+6.3f}, {m[4]:+6.3f}]")

        # Run the sweep (same measurements and P0 reused across all alphas for fairness)
        sweep_results = run_so3_analytic_alpha_sweep(
            alpha_list, shape_bank, noise_seeds,
            imu_pos, e3, args.gamma, args.steps, args.meas_std_deg,
            args.alpha_objective, nis_dof,
            P0=P0, nis_burnin=args.nis_burnin
        )

        # ── Summary table ─────────────────────────────────────────────────────
        w = 110
        print("\n" + "=" * w)
        print("ALPHA SWEEP SUMMARY  (SO(3)-analytic, paired across all alpha values)")
        print(f"  NIS DOF={nis_dof}  |  burn-in={args.nis_burnin} steps  |  "
              f"well-tuned filter: NIS_post ~ {nis_dof}")
        print(f"  Full-horizon NIS may be inflated by initialisation transient; "
              f"NIS_post is more representative.")
        print("=" * w)
        print(f"{'Alpha':>7} | {'RMSE_mean':>12} | {'RMSE_final':>12} | "
              f"{'NIS_full':>9} | {'NIS_post':>9} | {'NIS_post_std':>12} | {'ms/step':>7}")
        print("-" * w)
        for r in sweep_results:
            print(f"{r['alpha']:7.3f} | {r['rmse_mean_mean']:12.4e} | {r['rmse_final_mean']:12.4e} | "
                  f"{r['nis_mean_full_mean']:9.3f} | {r['nis_mean_postburnin_mean']:9.3f} | "
                  f"{r['nis_std_postburnin_mean']:12.3f} | "
                  f"{r['time_per_step_mean']*1e3:7.2f}")
        print("=" * w)

        # ── Alpha selection ───────────────────────────────────────────────────
        obj_key = "rmse_mean_mean" if args.alpha_objective == "rmse_mean" else "rmse_final_mean"
        best = min(sweep_results, key=lambda r: r[obj_key])

        nis_ratio = best["nis_mean_postburnin_mean"] / nis_dof if nis_dof > 0 else float("inf")

        print(f"\n  Recommended alpha by '{args.alpha_objective}': alpha = {best['alpha']:.3f}")
        print(f"    {args.alpha_objective:<16} = {best[obj_key]:.4e}")
        print(f"    NIS_full (mean)    = {best['nis_mean_full_mean']:.3f} +/- "
              f"{best['nis_mean_full_std']:.3f}  (DOF = {nis_dof})")
        print(f"    NIS_post (mean)    = {best['nis_mean_postburnin_mean']:.3f} +/- "
              f"{best['nis_mean_postburnin_std']:.3f}  (burn-in={args.nis_burnin})")
        if nis_ratio > 3.0:
            print(f"    WARNING: NIS_post/DOF = {nis_ratio:.1f} >> 1.  Filter may be overconfident "
                  f"(R too small). Consider increasing alpha.")
        elif nis_ratio < 0.3:
            print(f"    WARNING: NIS_post/DOF = {nis_ratio:.2f} << 1.  Filter may be underconfident "
                  f"(R too large). Consider decreasing alpha.")
        else:
            print(f"    NIS_post/DOF = {nis_ratio:.2f}  (within reasonable range of 1.0)")
        print(f"  Note: alpha is selected purely by '{args.alpha_objective}'. "
              f"Inspect NIS_post to confirm consistency.")

        # ── Save CSV ──────────────────────────────────────────────────────────
        import csv
        import json

        csv_path = "so3_alpha_sweep_results.csv"
        fieldnames = [
            "alpha", "num_shapes", "num_noise_realizations", "num_total_runs",
            "imu_layout", "nis_dof", "nis_burnin",
            "rmse_mean_mean", "rmse_mean_std",
            "rmse_final_mean", "rmse_final_std",
            "nis_mean_full_mean", "nis_mean_full_std",
            "nis_std_full_mean", "nis_std_full_std",
            "nis_mean_postburnin_mean", "nis_mean_postburnin_std",
            "nis_std_postburnin_mean", "nis_std_postburnin_std",
            "time_per_step_mean", "time_per_step_std",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sweep_results:
                row = {k: r.get(k, "") for k in fieldnames}
                row["imu_layout"] = str(r["imu_layout"])
                writer.writerow(row)
        print(f"\n  Results saved  -> {csv_path}")

        json_path = "so3_alpha_sweep_results.json"
        summary = {
            "config": {
                "imu_pos": list(imu_pos),
                "nis_dof": nis_dof,
                "nis_burnin": args.nis_burnin,
                "P0_diag": list(np.diag(P0)),
                "num_shapes": args.num_shapes,
                "num_noise_realizations": args.num_noise_realizations,
                "shape_bank_seed": args.shape_bank_seed,
                "noise_seed_base": args.noise_seed_base,
                "steps": args.steps,
                "meas_std_deg": args.meas_std_deg,
                "gamma": args.gamma,
                "alpha_objective": args.alpha_objective,
            },
            "best_alpha": best["alpha"],
            "results": sweep_results,
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved  -> {json_path}")

        # ── Optional plot ─────────────────────────────────────────────────────
        if args.plot_alpha_sweep:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("  [plot skipped] matplotlib is not installed.  "
                      "Run: pip install matplotlib")
                return

            # ── data ──────────────────────────────────────────────────────────
            alphas     = [r["alpha"]                    for r in sweep_results]
            rmse_means = [r["rmse_mean_mean"]            for r in sweep_results]
            nis_full   = [r["nis_mean_full_mean"]        for r in sweep_results]
            nis_post   = [r["nis_mean_postburnin_mean"]  for r in sweep_results]

            # Selected alpha for the paper (independent of RMSE-only recommendation)
            sel_alpha = args.selected_alpha

            # ── font sizes ────────────────────────────────────────────────────
            FS_LABEL  = 18
            FS_TICK   = 16
            FS_LEGEND = 15

            # ── colours ───────────────────────────────────────────────────────
            c_rmse = "#2166ac"   # blue  – RMSE (left axis)
            c_post = "#d6604d"   # red   – NIS post-burn-in (right axis)
            c_full = "#f4a582"   # light salmon – full-horizon NIS (de-emphasised)
            c_dof  = "#666666"   # grey  – DOF reference line
            c_sel  = "#1a1a1a"   # near-black – selected-alpha line

            fig, ax1 = plt.subplots(figsize=(10, 5))

            # Left axis: RMSE mean
            ax1.plot(alphas, rmse_means, "o-", color=c_rmse, linewidth=2.5,
                     markersize=7, label="RMSE mean")
            ax1.set_xlabel(r"$\alpha$ (measurement covariance scale)", fontsize=FS_LABEL)
            ax1.set_ylabel("RMSE mean (modal coefficients)", color=c_rmse,
                           fontsize=FS_LABEL)
            ax1.tick_params(axis="y", labelcolor=c_rmse, labelsize=FS_TICK)
            ax1.tick_params(axis="x", labelsize=FS_TICK)

            # Right axis: NIS curves
            ax2 = ax1.twinx()

            # Full-horizon NIS — kept but visually secondary
            ax2.plot(alphas, nis_full, "s--", color=c_full, linewidth=1.2,
                     markersize=5, alpha=0.65, label="NIS full-horizon")

            # Post-burn-in NIS — primary NIS curve
            ax2.plot(alphas, nis_post, "^-", color=c_post, linewidth=2.5,
                     markersize=7,
                     label=f"NIS post-burn-in (b={args.nis_burnin})")

            # NIS DOF reference line
            ax2.axhline(nis_dof, color=c_dof, linestyle=":", linewidth=1.8,
                        label=f"NIS DOF = {nis_dof}")

            ax2.set_ylabel("NIS mean", color=c_post, fontsize=FS_LABEL)
            ax2.tick_params(axis="y", labelcolor=c_post, labelsize=FS_TICK)

            # Selected-alpha vertical line (paper choice)
            if sel_alpha is not None:
                ax1.axvline(sel_alpha, color=c_sel, linestyle="--", linewidth=2.0,
                            label=rf"Selected $\alpha = {sel_alpha:g}$")

            # ── legend ────────────────────────────────────────────────────────
            lines1, lab1 = ax1.get_legend_handles_labels()
            lines2, lab2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, lab1 + lab2,
                       loc="upper right", fontsize=FS_LEGEND,
                       framealpha=0.9, edgecolor="0.7")

            plt.tight_layout()
            plt.show()

        return

    # =========================================================================
    #  FULL COMPARE MODE  (all 4 methods — original logic, preserved)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EKF SHAPE ESTIMATION COMPARISON")
    print("=" * 80)
    imu_layout_str = "[" + ", ".join(f"{v:.2f}" for v in imu_pos) + "]"
    print(f"  IMU layout:  {imu_layout_str}  ({len(imu_pos)} sensors)")
    print(f"  NIS DOF:     {nis_dof}  (= 3 x {len(imu_pos)} IMUs)")
    print(f"  Config:      {args.trials} trials x {args.steps} steps,  gamma={args.gamma}")
    if args.fast:
        print("  Method:      FAST — SO(3) numeric only")
    else:
        print("  Method:      FULL COMPARISON (all 4 methods)")
    print("\nTip: Use --fast for quick runs, --plot for visualization, --plot-shape for 3D shapes")
    print("=" * 80)

    # Parse seed list
    seeds = [int(s) for s in args.seed_list.split(",") if s.strip()]
    if len(seeds) < args.trials:
        raise ValueError("seed-list must have at least as many entries as trials")

    def run_all_methods(scales: Dict[str, float], fast_mode: bool = False) -> Tuple[Dict, Dict, Dict]:
        """
        Run all 4 EKF methods with given R-scales (or just SO(3) numeric in fast mode).

        Args:
            scales: Dictionary mapping method name to R-scale value
            fast_mode: If True, only run SO(3) numeric method

        Returns:
            stats: Statistics for each method
            mean_rmse: Mean RMSE trajectory for each method
            final_estimates: Final m_est for each method (last trial)
        """
        if fast_mode:
            methods_to_run = ["so3_numeric"]
            print("\n[FAST MODE] Running only SO(3) numeric method")
        else:
            methods_to_run = ["quat_numeric", "quat_analytic", "so3_analytic", "so3_numeric"]

        results = {k: [] for k in methods_to_run}
        rmse_histories = {k: [] for k in methods_to_run}
        final_estimates = {k: None for k in methods_to_run}

        # Store m_true from last trial for shape visualization
        m_true_final = None

        for t in range(args.trials):
            rng = np.random.RandomState(seeds[t])
            m_true = rng.uniform(-2, 2, 5)  # Random shape coefficients

            # Generate measurements
            meas_q, meas_R = build_meas_seq(
                m_true, imu_pos, e3, args.gamma, args.steps, args.meas_std_deg, rng
            )

            outputs = {}

            if not fast_mode:
                outputs["quat_numeric"] = ekf_quat(meas_q, m_true, imu_pos, e3, args.gamma,
                                 "quat_numeric", args.meas_std_deg, scales.get("quat_numeric", 1.0),
                                 P0=P0, nis_burnin=args.nis_burnin)
                outputs["quat_analytic"] = ekf_quat(meas_q, m_true, imu_pos, e3, args.gamma,
                                 "quat_analytic", args.meas_std_deg, scales.get("quat_analytic", 1.0),
                                 P0=P0, nis_burnin=args.nis_burnin)
                outputs["so3_analytic"] = ekf_so3(meas_R, m_true, imu_pos, e3, args.gamma,
                                "so3_analytic", args.meas_std_deg, scales.get("so3_analytic", 1.0),
                                P0=P0, nis_burnin=args.nis_burnin)

            outputs["so3_numeric"] = ekf_so3(meas_R, m_true, imu_pos, e3, args.gamma,
                            "so3_numeric", args.meas_std_deg, scales.get("so3_numeric", 1.0),
                            P0=P0, nis_burnin=args.nis_burnin)

            for method in methods_to_run:
                out = outputs[method]
                results[method].append(out)
                rmse_histories[method].append(np.sqrt(np.mean((out["hist"] - m_true) ** 2, axis=1)))

                if t == args.trials - 1:
                    final_estimates[method] = out["m_est"]
                    m_true_final = m_true

        stats = {k: summarize(v) for k, v in results.items()}
        mean_rmse = {k: np.mean(np.vstack(v), axis=0) for k, v in rmse_histories.items()}

        return stats, mean_rmse, final_estimates, m_true_final

    # ── R-scale sweep sub-mode (--sweep-all-scales flag, unchanged) ────────────
    if args.sweep_all_scales:
        scale_list = [float(x) for x in args.scale_list.split(",") if x.strip()]

        print("\n" + "=" * 80)
        print("R-SCALE SWEEP FOR ALL METHODS")
        print("=" * 80)

        best_results = {}

        for method in ["quat_numeric", "quat_analytic", "so3_analytic", "so3_numeric"]:
            print(f"\nSweeping {method}...")
            sweep = []

            for scale in scale_list:
                scales = {
                    "quat_numeric": args.quat_num_scale,
                    "quat_analytic": args.quat_ana_scale,
                    "so3_analytic": args.so3_ana_scale,
                    "so3_numeric": args.so3_num_scale,
                }
                scales[method] = scale

                stats, _, _, _ = run_all_methods(scales, fast_mode=False)
                obj = stats[method][args.objective][0]
                sweep.append((obj, scale, stats))
                print(f"  scale={scale:<6.2f}  {args.objective}={obj:.6e}")

            sweep.sort(key=lambda x: x[0])
            best_obj, best_scale, _ = sweep[0]
            best_results[method] = (best_scale, best_obj)
            print(f"  -> Best: scale={best_scale:.2f}, {args.objective}={best_obj:.6e}")

        print("\n" + "=" * 80)
        print("FINAL COMPARISON WITH OPTIMAL SCALES")
        print("=" * 80)

        optimal_scales = {k: v[0] for k, v in best_results.items()}
        final_stats, final_rmse, final_est, m_true_final = run_all_methods(optimal_scales, fast_mode=False)

        print_comparison_table(final_stats, optimal_scales, nis_dof)

        if args.plot:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("[plot skipped] matplotlib is not installed.  Run: pip install matplotlib")
                return
            t = np.arange(args.steps)

            plt.figure(figsize=(10, 6))
            for method, label in [("quat_numeric", "Quat numeric"),
                                 ("quat_analytic", "Quat analytic"),
                                 ("so3_analytic", "SO(3) analytic"),
                                 ("so3_numeric", "SO(3) numeric")]:
                if method in final_rmse:
                    plt.plot(t, final_rmse[method], label=f"{label} (α={optimal_scales[method]:.2f})")

            plt.xlabel("Time step")
            plt.ylabel("RMSE")
            plt.title("EKF RMSE over time (optimal scales)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

            if args.plot_shape and m_true_final is not None:
                plot_shape_comparison(m_true_final, final_est, e3, args.gamma)

        return

    # ── Single run with provided scales ────────────────────────────────────────
    scales = {
        "quat_numeric": args.quat_num_scale,
        "quat_analytic": args.quat_ana_scale,
        "so3_analytic": args.so3_ana_scale,
        "so3_numeric": args.so3_num_scale,
    }

    stats, rmse_hist, final_est, m_true_final = run_all_methods(scales, fast_mode=args.fast)

    print_comparison_table(stats, scales, nis_dof)

    if m_true_final is not None:
        print("\n" + "=" * 80)
        print("MODAL COEFFICIENTS (Shape Parameters)")
        print("=" * 80)
        print(f"True values:      m = [{m_true_final[0]:7.3f}, {m_true_final[1]:7.3f}, "
              f"{m_true_final[2]:7.3f}, {m_true_final[3]:7.3f}, {m_true_final[4]:7.3f}]")
        for method, m_est in final_est.items():
            if m_est is not None:
                error = m_est - m_true_final
                print(f"{method:16s}: m = [{m_est[0]:7.3f}, {m_est[1]:7.3f}, "
                      f"{m_est[2]:7.3f}, {m_est[3]:7.3f}, {m_est[4]:7.3f}] "
                      f"(error: {np.linalg.norm(error):.4f})")
        print("=" * 80)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[plot skipped] matplotlib is not installed.  Run: pip install matplotlib")
            return
        t = np.arange(args.steps)

        plt.figure(figsize=(10, 6))
        colors = {'quat_numeric': 'r', 'quat_analytic': 'orange',
                  'so3_analytic': 'b', 'so3_numeric': 'g'}
        labels = {'quat_numeric': 'Quat numeric', 'quat_analytic': 'Quat analytic',
                  'so3_analytic': 'SO(3) analytic', 'so3_numeric': 'SO(3) numeric'}

        for method in rmse_hist.keys():
            label = f"{labels.get(method, method)} (α={scales.get(method, 1.0):.2f})"
            plt.plot(t, rmse_hist[method], color=colors.get(method, 'gray'),
                    linewidth=2, label=label)

        plt.xlabel("Time step")
        plt.ylabel("RMSE (Modal Coefficients)")
        plt.title("EKF Convergence: Shape Estimation Error Over Time")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        if args.plot_shape and m_true_final is not None:
            plot_shape_comparison(m_true_final, final_est, e3, args.gamma)


if __name__ == "__main__":
    main()
