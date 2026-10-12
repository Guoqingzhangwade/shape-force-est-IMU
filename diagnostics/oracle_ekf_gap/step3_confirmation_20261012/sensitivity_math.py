"""Coordinate scaling and local derivatives for the bounded Step 2 diagnosis.

These helpers do not fit states, invert production wrenches, transform residual
rows, or read data. Callers must preserve the original least-squares row metric.
For m = S q, modal_scale gives ||dq||^2 = integral ||d kappa(s)||^2 ds.
The wrench derivative includes both sides of B(m) x(m) ~= y(m), including the
nonzero least-squares residual. It assumes all six wrench columns are retained.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag, solve_triangular


def modal_scale(vw, order):
    """Return Q and S with Q = blockdiag(Gram(order)) and S.T @ Q @ S = I.

    np.linalg.cholesky returns lower L with Q = L @ L.T; consequently
    q = L.T @ m and m = S @ q, S = solve(L.T, I). This is a curvature
    RMS coordinate choice, not an EKF covariance or an energy weighting.
    """
    if len(order) != 3:
        raise ValueError("Expected three polynomial orders")
    Q = block_diag(*(np.asarray(vw.gram_matrix(axis_order), dtype=float)
                     for axis_order in order))
    lower = np.linalg.cholesky(Q)
    S = solve_triangular(lower.T, np.eye(Q.shape[0]), lower=False)
    return Q, S


def least_squares_derivative(B, y, x, dB, dy):
    """Return (Jx, components) for a full-column-rank least-squares solution.

    Shapes are B=(nrow,6), y=(nrow,), x=(6,), dB=(nrow,6,8),
    dy=(nrow,8). dB and dy use the same eight input coordinates.

    With r=y-Bx and B.T r=0, differentiation gives
      (B.T B) dx = B.T (dy - dB x) + dB.T r.
    The implementation uses an SVD and never forms an inverse normal matrix.
    x must be the caller's existing solution under this same residual metric;
    this function does not replace or reweight that solution. A rank or cutoff
    transition invalidates this full-rank local derivative.
    """
    B, y, x = (np.asarray(value, dtype=float) for value in (B, y, x))
    dB, dy = (np.asarray(value, dtype=float) for value in (dB, dy))
    if B.ndim != 2 or B.shape[1] != 6 or B.shape[0] < 6:
        raise ValueError("Expected B shape (nrow, 6), with nrow >= 6")
    nrow = B.shape[0]
    if y.shape != (nrow,) or x.shape != (6,):
        raise ValueError("Expected y shape (nrow,) and x shape (6,)")
    if dB.shape != (nrow, 6, 8) or dy.shape != (nrow, 8):
        raise ValueError("Expected dB shape (nrow,6,8) and dy shape (nrow,8)")
    if not all(np.all(np.isfinite(value)) for value in (B, y, x, dB, dy)):
        raise ValueError("Least-squares derivative inputs must be finite")

    U, singular_values, Vt = np.linalg.svd(B, full_matrices=False)
    ratio = float(singular_values[-1] / singular_values[0]) if singular_values[0] > 0 else 0.0
    assert ratio > 1e-8, (
        "Full six-column derivative requires smallest/largest singular-value "
        f"ratio > 1e-8; got {ratio:.6g}"
    )
    V = Vt.T
    Bplus = (V / singular_values) @ U.T
    residual = y - B @ x
    db_term = Bplus @ dy
    dA_term = -Bplus @ np.einsum("ijk,j->ik", dB, x)
    residual_rhs = np.einsum("ijk,i->jk", dB, residual)
    residual_term = V @ ((Vt @ residual_rhs) / singular_values[:, None] ** 2)
    Jx = db_term + dA_term + residual_term
    components = {
        "db_term": db_term,
        "dA_term": dA_term,
        "residual_term": residual_term,
        "residual": residual,
        "stationarity_vector": B.T @ residual,
        "singular_values": singular_values,
        "smallest_largest_ratio": ratio,
    }
    return Jx, components


def directional_gains(Hwhite, Jx):
    """Return gains along deterministically signed measurement SVD directions.

    Hwhite=(9,8) maps unit curvature-RMS coordinates to whitened residuals;
    Jx=(6,8) maps those coordinates to a caller-defined scaled wrench.
    Columns of V are ordered strongest to weakest. Their largest-magnitude
    entry is positive (ties use NumPy's first index). Both U and V are flipped
    together so Hwhite = U @ diag(s) @ V.T remains valid.

    total_response_operator is Jx @ pinv(Hwhite), using an explicitly reported
    rcond=1e-15. It describes a local unregularized linear composition, not the
    EKF's gain or posterior covariance. Per-noise gains below the reported
    retained threshold are formal directional ratios, not operator gains.
    """
    Hwhite, Jx = (np.asarray(value, dtype=float) for value in (Hwhite, Jx))
    if Hwhite.shape != (9, 8) or Jx.shape != (6, 8):
        raise ValueError("Expected Hwhite shape (9,8) and Jx shape (6,8)")
    if not np.all(np.isfinite(Hwhite)) or not np.all(np.isfinite(Jx)):
        raise ValueError("Directional gain inputs must be finite")
    U, singular_values, Vt = np.linalg.svd(Hwhite, full_matrices=False)
    for index in range(len(singular_values)):
        pivot = int(np.argmax(np.abs(Vt[index])))
        if Vt[index, pivot] < 0:
            Vt[index] *= -1.0
            U[:, index] *= -1.0
    V = Vt.T
    wrench_gain = np.linalg.norm(Jx @ V, axis=0)
    per_noise_gain = np.full_like(singular_values, np.inf)
    np.divide(wrench_gain, singular_values, out=per_noise_gain,
              where=singular_values > 0)
    pinv_rcond = 1e-15
    retained = singular_values > pinv_rcond * singular_values[0]
    return {
        "U": U,
        "V": V,
        "singular_values": singular_values,
        "measurement_gain": singular_values.copy(),
        "wrench_gain": wrench_gain,
        "per_noise_gain": per_noise_gain,
        "measurement_pinv_rcond": pinv_rcond,
        "measurement_retained": retained,
        "measurement_retained_rank": int(np.count_nonzero(retained)),
        "total_response_operator": Jx @ np.linalg.pinv(Hwhite, rcond=pinv_rcond),
    }
