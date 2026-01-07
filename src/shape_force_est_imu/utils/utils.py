"""Common helper utilities (hat operator etc.)."""
from __future__ import annotations
import numpy as np
from numpy.linalg import norm

__all__ = ["hat", "vec", "unit", "block33_to_mat"]

def hat(v: np.ndarray) -> np.ndarray:
    """Skew‑symmetric matrix (× operator)."""
    x, y, z = v
    return np.array([[ 0, -z,  y],
                     [ z,  0, -x],
                     [-y,  x,  0]])

def vec(M: np.ndarray) -> np.ndarray:
    """Inverse of *hat* for 3 × 3 skew‑sym. matrix."""
    return np.array([M[2,1], M[0,2], M[1,0]])

def unit(v: np.ndarray) -> np.ndarray:
    """Return unit vector; zero stays zero."""
    n = norm(v)
    return v if n == 0 else v / n

def block33_to_mat(rows: list[np.ndarray]) -> np.ndarray:
    """Stack three row‑vectors of length 3 into 3 × 3 matrix."""
    return np.vstack(rows)