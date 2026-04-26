#!/usr/bin/env python3
"""
Shared helpers for the new wrench-observability and estimator-comparison studies.

These utilities are intentionally separate from the existing Step-4 scripts so
the original workflow and saved results remain untouched.

Conventions
-----------
- Wrench vector ordering is always:
      F = [Mx, My, Mz, Fx, Fy, Fz]
  i.e. [moment; force].
- `J_Vb_m` is the 6 x n_m body Jacobian at the tip.
- `J_qm = dq/dm = -dell/dm` is the tendon pull/shortening
  Jacobian for q = ell_0 - ell.
- The virtual-work wrench map is
      b_w = gradU(m) - J_qm.T @ tau = J_Vb_m.T @ F_b
  so the matrix of interest for observability is `J_Vb_m.T` with shape
  (n_m, 6).
- GT wrench components in the Kirchhoff dataset are stored in world axes as
  separate arrays `f_ext` and `l_ext`; we combine them into the same ordering
  `[moment_world; force_world]` when computing metrics.

Important study note
--------------------
The observability-vs-order study evaluates `J_Vb_m.T` on modal states obtained
by projecting each saved Kirchhoff-rod case onto the requested polynomial modal
basis using a least-squares fit to curvature derived from the saved GT frames.
This gives a fair, order-dependent modal state for the same underlying shape
without modifying the current EKF implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, NamedTuple, Sequence, Tuple

import numpy as np
from scipy.linalg import block_diag

# NumPy < 1.20 compatibility for existing virtual-work helpers.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

from virtual_work import (
    body_jacobian_at_s,
    elastic_energy_gradient,
    generalized_modal_load,
    gram_matrix,
    pull_jacobian,
    total_params,
)


# ---------------------------------------------------------------------------
# Shared rod / tendon parameters
# ---------------------------------------------------------------------------

L_DEFAULT = 0.1
E_DEFAULT = 60e9
NU_DEFAULT = 0.3
G_DEFAULT = E_DEFAULT / (2.0 * (1.0 + NU_DEFAULT))
R_BACKBONE_DEFAULT = 5e-4
I_BACKBONE_DEFAULT = np.pi * R_BACKBONE_DEFAULT**4 / 4.0
EIX_DEFAULT = E_DEFAULT * I_BACKBONE_DEFAULT
EIY_DEFAULT = EIX_DEFAULT
GJ_DEFAULT = G_DEFAULT * 2.0 * I_BACKBONE_DEFAULT

R_TENDON_DEFAULT = 0.008
R_LIST_DEFAULT = [
    np.array([R_TENDON_DEFAULT, 0.0, 0.0]),
    np.array([0.0, R_TENDON_DEFAULT, 0.0]),
    np.array([-R_TENDON_DEFAULT, 0.0, 0.0]),
    np.array([0.0, -R_TENDON_DEFAULT, 0.0]),
]


class OrderConfig(NamedTuple):
    order_x: int
    order_y: int
    order_z: int


BASELINE_ORDER = (1, 1, 0)
DEFAULT_ORDER_CONFIGS = [
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 0),
    (2, 2, 0),
    (2, 2, 1),
    (3, 3, 1),
]

COMPONENT_LABELS = ("Mx", "My", "Mz", "Fx", "Fy", "Fz")


# ---------------------------------------------------------------------------
# Formatting / parsing
# ---------------------------------------------------------------------------

def order_label(order_cfg: Tuple[int, int, int]) -> str:
    ox, oy, oz = order_cfg
    return f"({ox},{oy},{oz})"


def order_slug(order_cfg: Tuple[int, int, int]) -> str:
    ox, oy, oz = order_cfg
    return f"ox{ox}_oy{oy}_oz{oz}"


def parse_order_configs(order_text: str) -> List[Tuple[int, int, int]]:
    """
    Parse semicolon-separated order tuples, e.g.
        "1,1,0;1,1,1;2,2,0"
    """
    cfgs: List[Tuple[int, int, int]] = []
    for chunk in order_text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [int(v.strip()) for v in chunk.split(",")]
        if len(vals) != 3:
            raise ValueError(f"Expected 3 integers per order tuple, got: {chunk}")
        cfgs.append((vals[0], vals[1], vals[2]))
    if not cfgs:
        raise ValueError("No modal-order configurations parsed from --orders.")
    return cfgs


# ---------------------------------------------------------------------------
# GT-frame -> modal projection
# ---------------------------------------------------------------------------

def build_transform_stack(
    positions: np.ndarray,
    orientations: np.ndarray,
) -> np.ndarray:
    """
    Convert saved GT arrays into a homogeneous transform stack.

    Parameters
    ----------
    positions     : (M, 3)
    orientations  : (M, 3, 3)
    """
    M = positions.shape[0]
    T = np.zeros((M, 4, 4), dtype=float)
    T[:, :3, :3] = orientations
    T[:, :3, 3] = positions
    T[:, 3, 3] = 1.0
    return T


def rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    cos_val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_val))
    if theta < 1e-12:
        return np.zeros(3)
    w_hat = (R - R.T) / (2.0 * np.sin(theta))
    return theta * np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])


def estimate_modal_from_gt_frames(
    T_obs: np.ndarray,
    s_values: np.ndarray,
    order_x: int,
    order_y: int,
    order_z: int,
) -> np.ndarray:
    """
    Least-squares modal fit from discrete GT frames.

    This mirrors the helper already used in `scripts/force_est/force_est_curve_fit.py`.
    We estimate discrete curvature from neighbouring GT orientations and then
    fit each curvature axis independently to the requested monomial basis.
    """
    s_mid = 0.5 * (s_values[:-1] + s_values[1:])
    ds = float(s_values[1] - s_values[0])
    kappa = np.zeros((len(s_mid), 3), dtype=float)

    for idx in range(len(s_mid)):
        R_i = T_obs[idx, :3, :3]
        R_j = T_obs[idx + 1, :3, :3]
        rotvec = rotation_matrix_to_rotvec(R_i.T @ R_j)
        kappa[idx] = rotvec / ds

    def _fit_axis(samples: np.ndarray, order: int) -> np.ndarray:
        Phi = np.vstack([s_mid**power for power in range(order + 1)]).T
        return np.linalg.lstsq(Phi, samples, rcond=None)[0]

    mx = _fit_axis(kappa[:, 0], order_x)
    my = _fit_axis(kappa[:, 1], order_y)
    mz = _fit_axis(kappa[:, 2], order_z)
    return np.hstack([mx, my, mz])


# ---------------------------------------------------------------------------
# Virtual-work assembly
# ---------------------------------------------------------------------------

@dataclass
class VirtualWorkTerms:
    order_cfg: Tuple[int, int, int]
    m: np.ndarray
    tau: np.ndarray
    gradU: np.ndarray
    J_qm: np.ndarray
    J_vbm: np.ndarray
    T_tip: np.ndarray
    A: np.ndarray
    b: np.ndarray


def build_elastic_hessian(
    order_x: int,
    order_y: int,
    order_z: int,
    length_m: float = L_DEFAULT,
    EIx: float = EIX_DEFAULT,
    EIy: float = EIY_DEFAULT,
    GJ: float = GJ_DEFAULT,
) -> np.ndarray:
    return block_diag(
        (EIx / length_m) * gram_matrix(order_x),
        (EIy / length_m) * gram_matrix(order_y),
        (GJ / length_m) * gram_matrix(order_z),
    )


def build_virtual_work_terms(
    m: np.ndarray,
    tau: np.ndarray,
    order_cfg: Tuple[int, int, int],
    gamma: int,
    length_m: float = L_DEFAULT,
    EIx: float = EIX_DEFAULT,
    EIy: float = EIY_DEFAULT,
    GJ: float = GJ_DEFAULT,
    r_list: Sequence[np.ndarray] = R_LIST_DEFAULT,
) -> VirtualWorkTerms:
    order_x, order_y, order_z = order_cfg
    gradU = elastic_energy_gradient(
        m, EIx, EIy, GJ, length_m, order_x, order_y, order_z
    )
    J_qm = pull_jacobian(m, r_list, length_m, order_x, order_y, order_z)
    J_vbm, T_tip = body_jacobian_at_s(m, 1.0, gamma, length_m, order_x, order_y, order_z)
    A = J_vbm.T
    b = generalized_modal_load(gradU, J_qm, tau)
    return VirtualWorkTerms(
        order_cfg=order_cfg,
        m=m,
        tau=tau,
        gradU=gradU,
        J_qm=J_qm,
        J_vbm=J_vbm,
        T_tip=T_tip,
        A=A,
        b=b,
    )


def generalized_load_from_state(
    m: np.ndarray,
    tau: np.ndarray,
    order_cfg: OrderConfig,
    length_m: float = L_DEFAULT,
    EIx: float = EIX_DEFAULT,
    EIy: float = EIY_DEFAULT,
    GJ: float = GJ_DEFAULT,
    r_list: list[np.ndarray] | None = None,
) -> np.ndarray:
    """
    Compute the generalized modal load attributed to the external wrench:

        b_w(m, tau) = gradU(m) - J_qm(m).T @ tau,

    where J_qm = dq/dm is the tendon pull/shortening Jacobian.
    """
    order_x, order_y, order_z = order_cfg
    routing = R_LIST_DEFAULT if r_list is None else r_list
    m = np.asarray(m, dtype=float)
    tau = np.asarray(tau, dtype=float)
    gradU = elastic_energy_gradient(
        m, EIx, EIy, GJ, length_m, order_x, order_y, order_z
    )
    J_qm = pull_jacobian(m, routing, length_m, order_x, order_y, order_z)
    return generalized_modal_load(gradU, J_qm, tau)


def finite_difference_B_m_for_generalized_load(
    m: np.ndarray,
    tau: np.ndarray,
    order_cfg: OrderConfig,
    eps: float = 1e-6,
    length_m: float = L_DEFAULT,
    EIx: float = EIX_DEFAULT,
    EIy: float = EIY_DEFAULT,
    GJ: float = GJ_DEFAULT,
    r_list: list[np.ndarray] | None = None,
) -> np.ndarray:
    """
    Central-difference approximation of:

        B_m = d b_w / d m.

    The output shape is (n_params, n_params).
    """
    m = np.asarray(m, dtype=float)
    n_params = len(m)
    B_m = np.zeros((n_params, n_params), dtype=float)
    for j in range(n_params):
        dm = np.zeros_like(m)
        dm[j] = eps
        b_plus = generalized_load_from_state(
            m + dm, tau, order_cfg, length_m, EIx, EIy, GJ, r_list
        )
        b_minus = generalized_load_from_state(
            m - dm, tau, order_cfg, length_m, EIx, EIy, GJ, r_list
        )
        B_m[:, j] = (b_plus - b_minus) / (2.0 * eps)
    return B_m


def propagate_generalized_load_covariance(
    m: np.ndarray,
    tau: np.ndarray,
    P_modal: np.ndarray,
    R_tau: np.ndarray,
    order_cfg: OrderConfig,
    Sigma_b_model: np.ndarray | None = None,
    eps: float = 1e-6,
    length_m: float = L_DEFAULT,
    EIx: float = EIX_DEFAULT,
    EIy: float = EIY_DEFAULT,
    GJ: float = GJ_DEFAULT,
    r_list: list[np.ndarray] | None = None,
) -> np.ndarray:
    """
    Propagate modal-state and tendon-tension uncertainty to the
    generalized load residual:

        Sigma_b = B_m P_modal B_m.T
                + B_tau R_tau B_tau.T
                + Sigma_b_model

    with:
        B_tau = -J_qm.T.
    """
    order_x, order_y, order_z = order_cfg
    routing = R_LIST_DEFAULT if r_list is None else r_list
    m = np.asarray(m, dtype=float)
    tau = np.asarray(tau, dtype=float)
    P_modal = np.asarray(P_modal, dtype=float)
    R_tau = np.asarray(R_tau, dtype=float)

    n_params = total_params(order_x, order_y, order_z)
    if m.shape != (n_params,):
        raise ValueError(f"Expected m shape {(n_params,)}, got {m.shape}.")
    if P_modal.shape != (n_params, n_params):
        raise ValueError(
            f"Expected P_modal shape {(n_params, n_params)}, got {P_modal.shape}."
        )
    if len(routing) != len(tau):
        raise ValueError(
            f"Expected tau length {len(routing)} for {len(routing)} tendons, "
            f"got {len(tau)}."
        )
    if R_tau.shape != (len(tau), len(tau)):
        raise ValueError(
            f"Expected R_tau shape {(len(tau), len(tau))}, got {R_tau.shape}."
        )

    B_m = finite_difference_B_m_for_generalized_load(
        m, tau, order_cfg, eps, length_m, EIx, EIy, GJ, routing
    )
    J_qm = pull_jacobian(m, routing, length_m, order_x, order_y, order_z)
    B_tau = -J_qm.T
    Sigma_b = B_m @ P_modal @ B_m.T + B_tau @ R_tau @ B_tau.T

    if Sigma_b_model is not None:
        Sigma_b_model = np.asarray(Sigma_b_model, dtype=float)
        if Sigma_b_model.shape != (n_params, n_params):
            raise ValueError(
                "Expected Sigma_b_model shape "
                f"{(n_params, n_params)}, got {Sigma_b_model.shape}."
            )
        Sigma_b = Sigma_b + Sigma_b_model

    return 0.5 * (Sigma_b + Sigma_b.T)


# ---------------------------------------------------------------------------
# Wrench transforms / solvers
# ---------------------------------------------------------------------------

def wrench_rotation_block(R: np.ndarray) -> np.ndarray:
    return block_diag(R, R)


def body_wrench_to_world(F_body: np.ndarray, R_tip: np.ndarray) -> np.ndarray:
    return wrench_rotation_block(R_tip) @ F_body


def world_wrench_to_body(F_world: np.ndarray, R_tip: np.ndarray) -> np.ndarray:
    return wrench_rotation_block(R_tip.T) @ F_world


def direct_wrench_body_from_terms(
    terms: VirtualWorkTerms,
    rcond: float = 1e-8,
) -> np.ndarray:
    return np.linalg.pinv(terms.A, rcond=rcond) @ terms.b


def moment_only_wrench_body_from_terms(
    terms: VirtualWorkTerms,
    rcond: float = 1e-8,
) -> np.ndarray:
    """
    Observability-informed constrained estimator.

    Based on the current 5D model's singular-vector structure, the strongly
    observable subspace is moment-dominant while weak/null directions lie almost
    entirely in the force subspace. This estimator therefore solves only for
    the 3 body-tip moment components and enforces zero force.
    """
    A_moment = terms.A[:, :3]
    M_body = np.linalg.pinv(A_moment, rcond=rcond) @ terms.b
    return np.hstack([M_body, np.zeros(3)])


def truncated_svd_wrench_body_from_terms(
    terms: VirtualWorkTerms,
    rel_sigma_cut: float,
    rcond: float = 1e-12,
) -> Tuple[np.ndarray, int]:
    U, svals, Vt = np.linalg.svd(terms.A, full_matrices=False)
    if len(svals) == 0 or svals[0] <= rcond:
        return np.zeros(6), 0
    keep = (svals / svals[0]) > rel_sigma_cut
    if not np.any(keep):
        keep[0] = True
    U_r = U[:, keep]
    s_r = svals[keep]
    V_r = Vt[keep].T
    F_body = V_r @ ((U_r.T @ terms.b) / s_r)
    return F_body, int(np.sum(keep))


def propagate_wrench_covariance_body(
    terms: VirtualWorkTerms,
    P_modal: np.ndarray,
    sigma_model_force: float,
    sigma_model_moment: float,
    rcond: float = 1e-8,
    length_m: float = L_DEFAULT,
    EIx: float = EIX_DEFAULT,
    EIy: float = EIY_DEFAULT,
    GJ: float = GJ_DEFAULT,
) -> np.ndarray:
    """
    Legacy/post-pseudoinverse covariance diagnostic.

    The manuscript MAP estimator should use
    propagate_generalized_load_covariance() instead.
    """
    order_x, order_y, order_z = terms.order_cfg
    H_U = build_elastic_hessian(order_x, order_y, order_z, length_m, EIx, EIy, GJ)
    Sigma_bw = H_U @ P_modal @ H_U.T
    A_pinv = np.linalg.pinv(terms.A, rcond=rcond)
    Sigma_Fb = A_pinv @ Sigma_bw @ A_pinv.T
    Sigma_model = np.diag(
        [sigma_model_moment**2] * 3 + [sigma_model_force**2] * 3
    )
    return Sigma_Fb + Sigma_model


def recursive_map_update(
    mean_meas: np.ndarray,
    cov_meas: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    jitter: float = 1e-12,
) -> np.ndarray:
    """
    Closed-form recursive MAP fusion:

      argmin ||F - mean_meas||^2_{cov_meas^{-1}} + ||F - prior_mean||^2_{prior_cov^{-1}}

    This is a post-pseudoinverse diagnostic smoother. The manuscript MAP
    estimator should use the residual-level load-subspace formulation instead.
    """
    cov_meas_reg = cov_meas + jitter * np.eye(cov_meas.shape[0])
    prior_cov_reg = prior_cov + jitter * np.eye(prior_cov.shape[0])
    cov_meas_inv = np.linalg.inv(cov_meas_reg)
    prior_cov_inv = np.linalg.inv(prior_cov_reg)
    lhs = cov_meas_inv + prior_cov_inv
    rhs = cov_meas_inv @ mean_meas + prior_cov_inv @ prior_mean
    return np.linalg.solve(lhs, rhs)


# ---------------------------------------------------------------------------
# SVD / observability diagnostics
# ---------------------------------------------------------------------------

@dataclass
class SvdDiagnostics:
    singular_values: np.ndarray
    numerical_rank: int
    effective_rank: int
    cond_nonzero: float
    smallest_nonzero_sigma: float
    smallest_retained_sigma_effective: float
    weakest_right_vector: np.ndarray


def svd_diagnostics(
    A: np.ndarray,
    numerical_rank_rel_tol: float = 1e-6,
    effective_rank_rel_tol: float = 1e-2,
    abs_floor: float = 1e-12,
) -> SvdDiagnostics:
    """
    Analyze singular spectrum of the modal wrench map A = J_Vb_m.T.

    `numerical_rank` answers whether the matrix is algebraically full rank.
    `effective_rank` uses a looser threshold to expose practically weak
    directions that are technically nonzero but poorly conditioned.
    """
    U, svals, Vt = np.linalg.svd(A, full_matrices=True)
    svals = np.asarray(svals, dtype=float)
    s_padded = np.zeros(6, dtype=float)
    s_padded[: len(svals)] = svals

    if len(svals) == 0 or svals[0] <= abs_floor:
        return SvdDiagnostics(
            singular_values=s_padded,
            numerical_rank=0,
            effective_rank=0,
            cond_nonzero=np.inf,
            smallest_nonzero_sigma=0.0,
            smallest_retained_sigma_effective=0.0,
            weakest_right_vector=np.zeros(6),
        )

    rel = svals / svals[0]
    numerical_rank = int(np.sum(rel > numerical_rank_rel_tol))
    effective_rank = int(np.sum(rel > effective_rank_rel_tol))
    smallest_nonzero = float(svals[numerical_rank - 1]) if numerical_rank > 0 else 0.0
    smallest_effective = float(svals[effective_rank - 1]) if effective_rank > 0 else 0.0
    cond_nonzero = float(svals[0] / max(smallest_nonzero, abs_floor))

    if numerical_rank < 6:
        weakest = Vt[-1]
    else:
        weakest = Vt[len(svals) - 1]

    return SvdDiagnostics(
        singular_values=s_padded,
        numerical_rank=numerical_rank,
        effective_rank=effective_rank,
        cond_nonzero=cond_nonzero,
        smallest_nonzero_sigma=smallest_nonzero,
        smallest_retained_sigma_effective=smallest_effective,
        weakest_right_vector=np.asarray(weakest, dtype=float),
    )


# ---------------------------------------------------------------------------
# Wrench metrics
# ---------------------------------------------------------------------------

def _angle_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    cos_val = np.clip(vec_a @ vec_b / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def wrench_metrics_from_world6(
    F_est_world: np.ndarray,
    F_gt_world: np.ndarray,
    eps: float = 1e-12,
    estimate_force: bool = True,
    estimate_moment: bool = True,
) -> dict:
    """
    Compute requested force/moment metrics from world-tip wrench vectors.

    If a method intentionally does not estimate force or moment, the
    corresponding metrics are set to NaN so they do not contaminate summaries.
    """
    M_est = F_est_world[:3]
    F_est = F_est_world[3:]
    M_gt = F_gt_world[:3]
    F_gt = F_gt_world[3:]

    force_gt_norm = float(np.linalg.norm(F_gt))
    moment_gt_norm = float(np.linalg.norm(M_gt))
    force_est_norm = float(np.linalg.norm(F_est))
    moment_est_norm = float(np.linalg.norm(M_est))

    out = {
        "force_gt_norm_N": force_gt_norm,
        "moment_gt_norm_Nm": moment_gt_norm,
        "force_est_norm_N": force_est_norm if estimate_force else float("nan"),
        "moment_est_norm_Nm": moment_est_norm if estimate_moment else float("nan"),
    }

    if estimate_force:
        force_err = float(np.linalg.norm(F_est - F_gt))
        out.update(
            {
                "force_err_N": force_err,
                "force_norm_error": force_err / (force_gt_norm + eps),
                "force_mag_rel_error": abs(force_est_norm - force_gt_norm)
                / (force_gt_norm + eps),
                "force_dir_err_deg": _angle_deg(F_est, F_gt),
            }
        )
    else:
        out.update(
            {
                "force_err_N": float("nan"),
                "force_norm_error": float("nan"),
                "force_mag_rel_error": float("nan"),
                "force_dir_err_deg": float("nan"),
            }
        )

    if estimate_moment:
        moment_err = float(np.linalg.norm(M_est - M_gt))
        out.update(
            {
                "moment_err_Nm": moment_err,
                "moment_norm_error": moment_err / (moment_gt_norm + eps),
                "moment_mag_rel_error": abs(moment_est_norm - moment_gt_norm)
                / (moment_gt_norm + eps),
                "moment_dir_err_deg": _angle_deg(M_est, M_gt),
            }
        )
    else:
        out.update(
            {
                "moment_err_Nm": float("nan"),
                "moment_norm_error": float("nan"),
                "moment_mag_rel_error": float("nan"),
                "moment_dir_err_deg": float("nan"),
            }
        )

    return out
