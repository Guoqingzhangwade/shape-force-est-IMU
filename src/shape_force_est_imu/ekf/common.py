"""Shared math helpers for EKF scripts."""

from __future__ import annotations

import numpy as np


def skew(u: np.ndarray) -> np.ndarray:
    """Return 3x3 skew-symmetric matrix from 3-vector."""
    return np.array(
        [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]]
    )
