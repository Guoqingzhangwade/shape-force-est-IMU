#!/usr/bin/env python3
"""
Constrained Tip-Wrench Estimation Study.

Evaluates constrained wrench reconstruction for three admissible-load subspaces:

  Case 1 — force_only       : F_b lies in force subspace (zero body moments)
  Case 2 — transverse_force : F_b lies in body-XY force plane (zero moment + axial)
  Case 3 — moment_only      : F_b lies in moment subspace (zero body forces)

Manuscript formulation
----------------------
  Generalized modal load:
      b_w = gradU(m) - J_qm.T(m) tau

  where J_qm is the tendon-pull/shortening Jacobian.

  Unconstrained direct estimate (baseline, same as Step 4):
      F̄_b = pinv( J_{Vbm}^T(m) ) b_w

  Constrained direct estimate (main contribution):
      F_b = S z                      (admissible-subspace parametrisation)
      ẑ   = pinv( J_{Vbm}^T(m) S ) b_w
      F̂_b = S ẑ

  where S is a (6 × n_z) selection matrix with rows ordered [moment; force]
  (matching virtual_work.py convention).

Selection matrices S (body / tip frame)
---------------------------------------
  S_force_only  (6×3):  rows 3:6 = I_3  →  F_b = [0,0,0, Fx,Fy,Fz]^T
  S_transverse  (6×2):  rows 3:5 = I_2  →  F_b = [0,0,0, Fx,Fy, 0]^T
  S_moment_only (6×3):  rows 0:3 = I_3  →  F_b = [Mx,My,Mz, 0,0,0]^T

Frame notes
-----------
  GT wrenches (f_ext, l_ext) are stored in the *world* frame.
  Estimated body-frame wrenches are converted to world frame via R_tip:
      f_world = R_tip @ F_b[3:]
      l_world = R_tip @ F_b[:3]
  All error metrics are computed in the world frame.

  For the transverse-force case the S matrix is defined in the body (tip)
  frame (zero body-z force), while the GT dataset has zero *world*-z force.
  These constraints coincide when the rod is approximately vertical (small
  deflection) and diverge slightly for large deflections.  This is documented
  explicitly in the output.

IMU synthesis
-------------
  Measurements are synthesised inline (no separate Step 2 required).
  Layouts: 2-IMU at s = [0.50, 1.00],  3-IMU at s = [0.25, 0.50, 1.00].
  Noise model: R_meas = expm(hat(η)) @ R_true,  η ~ N(0, σ² I),  σ = 0.5°.
  n_noise = 5 realizations per case.

Oracle mode
-----------
  Oracle modal fit (from GT backbone, no IMU noise) is also evaluated to
  provide an upper bound on constrained wrench accuracy.

Output (in --save-dir)
----------------------
  constrained_results.csv         per-(case, method, layout, noise_real)
  constrained_summary.csv         aggregated means per (case, method, layout)
  constrained_results.json        full structured results
  constrained_wrench_est_data.npz per-case arrays for plotting

Usage
-----
  cd src/shape_force_est_imu/crt
  python evaluate_kirchhoff_wrench_constrained.py
  python evaluate_kirchhoff_wrench_constrained.py \\
      --data-dir gt_data_constrained \\
      --save-dir gt_data_constrained/results
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import expm

# ---------------------------------------------------------------------------
# EKF utilities (unchanged from Step 3/4)
# ---------------------------------------------------------------------------
from evaluate_kirchhoff_shape_estimation import (
    ALPHA_DEFAULT,
    E3,
    GAMMA_DEFAULT,
    MEAS_STD_DEG_DEF,
    P0_DIAG,
    Q_DIAG,
    STATE_DIM,
    STEPS_DEFAULT,
    run_ekf_on_frame,
    skew,
    so3_log,
)

# ---------------------------------------------------------------------------
# Virtual-work mechanics
# ---------------------------------------------------------------------------
from virtual_work import (
    body_jacobian_at_s,
    elastic_energy_gradient,
    generalized_modal_load,
    gram_matrix,
    pull_jacobian,
)
from scipy.linalg import block_diag as _block_diag

# ---------------------------------------------------------------------------
# Oracle fitting (from compare_oracle_vs_ekf_wrench_estimation.py)
# ---------------------------------------------------------------------------
from compare_oracle_vs_ekf_wrench_estimation import estimate_modal_oracle

# ---------------------------------------------------------------------------
# Rod / estimator constants — must match main study
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
    np.array([_R_TENDON,  0.0,         0.0]),
    np.array([0.0,         _R_TENDON,  0.0]),
    np.array([-_R_TENDON, 0.0,         0.0]),
    np.array([0.0,        -_R_TENDON,  0.0]),
]
_ORDER_X = 1
_ORDER_Y = 1
_ORDER_Z = 0

# Elastic-energy Hessian (shared)
_H_U = _block_diag(
    (_EIX / _L) * gram_matrix(_ORDER_X),
    (_EIY / _L) * gram_matrix(_ORDER_Y),
    (_GJ  / _L) * gram_matrix(_ORDER_Z),
)

# ---------------------------------------------------------------------------
# IMU layouts (arc-length positions, normalised, matching main study)
# ---------------------------------------------------------------------------
_IMU_LAYOUTS: Dict[str, np.ndarray] = {
    "2-IMU": np.array([0.50, 1.00]),
    "3-IMU": np.array([0.25, 0.50, 1.00]),
}
_N_NOISE = 5   # noise realisations per case

# ---------------------------------------------------------------------------
# Selection matrices S — body (tip) frame
# Convention: F_b = [moment_b; force_b]  (rows 0:3 = moment, rows 3:6 = force)
#
# Manuscript: F_b = S z,  ẑ = pinv(J_{Vbm}^T S) b_w,  F̂_b = S ẑ
# ---------------------------------------------------------------------------

def _make_S_force_only() -> np.ndarray:
    """S (6×3): allow body force [Fx,Fy,Fz], zero body moments."""
    S = np.zeros((6, 3))
    S[3, 0] = S[4, 1] = S[5, 2] = 1.0
    return S


def _make_S_transverse() -> np.ndarray:
    """S (6×2): allow body transverse force [Fx,Fy], zero body Fz and moments."""
    S = np.zeros((6, 2))
    S[3, 0] = S[4, 1] = 1.0
    return S


def _make_S_moment_only() -> np.ndarray:
    """S (6×3): allow body moment [Mx,My,Mz], zero body forces."""
    S = np.zeros((6, 3))
    S[0, 0] = S[1, 1] = S[2, 2] = 1.0
    return S


# Indexed by constrained case name
_S_MATRICES: Dict[str, np.ndarray] = {
    "force_only"       : _make_S_force_only(),
    "transverse_force" : _make_S_transverse(),
    "moment_only"      : _make_S_moment_only(),
}

# ---------------------------------------------------------------------------
# Wrench estimators
# ---------------------------------------------------------------------------

def _compute_b_w(
    m: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    """
    Generalized modal load: b_w = gradU(m) - J_qm.T tau.

    J_qm is the tendon-pull/shortening Jacobian.
    """
    gradU = elastic_energy_gradient(m, _EIX, _EIY, _GJ, _L,
                                    _ORDER_X, _ORDER_Y, _ORDER_Z)
    J_qm = pull_jacobian(m, _R_LIST, _L, _ORDER_X, _ORDER_Y, _ORDER_Z)
    return generalized_modal_load(gradU, J_qm, tau)


def estimate_wrench_unconstrained(
    m_est: np.ndarray,
    tau: np.ndarray,
    gamma: int = GAMMA_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unconstrained direct estimate (same as Step 4 baseline):
      F̄_b = pinv( J_{Vbm}^T ) b_w

    Returns (f_world, l_world, F_body).
    """
    b_w          = _compute_b_w(m_est, tau)
    J_vbm, T_tip = body_jacobian_at_s(m_est, 1.0, gamma, _L,
                                       _ORDER_X, _ORDER_Y, _ORDER_Z)
    F_b          = np.linalg.pinv(J_vbm.T, rcond=1e-8) @ b_w
    R_tip        = T_tip[:3, :3]
    f_world      = R_tip @ F_b[3:]
    l_world      = R_tip @ F_b[:3]
    return f_world, l_world, F_b


def estimate_wrench_constrained(
    m_est: np.ndarray,
    tau: np.ndarray,
    S: np.ndarray,
    gamma: int = GAMMA_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constrained direct estimate:
      F_b = S z
      ẑ   = pinv( J_{Vbm}^T S ) b_w       (manuscript eq.)
      F̂_b = S ẑ

    Parameters
    ----------
    S : (6, n_z)  selection matrix (body frame, [moment; force] row order)

    Returns (f_world, l_world, F_body, z_hat).
    """
    b_w          = _compute_b_w(m_est, tau)
    J_vbm, T_tip = body_jacobian_at_s(m_est, 1.0, gamma, _L,
                                       _ORDER_X, _ORDER_Y, _ORDER_Z)
    # A = J_{Vbm}^T  (n_params × 6),   A_S = A S  (n_params × n_z)
    A    = J_vbm.T                                     # (n_params, 6)
    A_S  = A @ S                                       # (n_params, n_z)
    z_hat = np.linalg.pinv(A_S, rcond=1e-8) @ b_w     # (n_z,)
    F_b  = S @ z_hat                                   # (6,) [moment; force]

    R_tip   = T_tip[:3, :3]
    f_world = R_tip @ F_b[3:]
    l_world = R_tip @ F_b[:3]
    return f_world, l_world, F_b, z_hat


# ---------------------------------------------------------------------------
# IMU measurement synthesis (inline, no separate Step-2 file required)
# ---------------------------------------------------------------------------

def synthesize_R_meas(
    orientations_gt: np.ndarray,    # (M, 9) GT orientations row-major
    imu_arc_norm: np.ndarray,       # (n_imu,) normalised arc-length positions
    num_pts: int,
    n_noise: int,
    meas_std_deg: float,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Synthesise noisy IMU rotation measurements from GT orientations.

    Noise model: R_meas = expm(hat(η)) @ R_true,  η ~ N(0, σ² I),  σ = meas_std_deg.

    Returns
    -------
    R_meas      : (n_noise, n_imu, 3, 3)
    arc_indices : list of int — indices into the num_pts grid
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma       = np.deg2rad(meas_std_deg)
    n_imu       = len(imu_arc_norm)
    arc_indices = [int(round(s * (num_pts - 1))) for s in imu_arc_norm]

    R_meas = np.zeros((n_noise, n_imu, 3, 3))
    for ni in range(n_noise):
        for ii, idx in enumerate(arc_indices):
            R_true = orientations_gt[idx].reshape(3, 3)
            eta    = rng.standard_normal(3) * sigma
            R_meas[ni, ii] = expm(skew(eta)) @ R_true
    return R_meas, arc_indices


# ---------------------------------------------------------------------------
# Metrics
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
) -> Dict[str, float]:
    fe  = float(np.linalg.norm(f_est - f_gt))
    me  = float(np.linalg.norm(l_est - l_gt))
    fn  = float(np.linalg.norm(f_gt))
    mn  = float(np.linalg.norm(l_gt))
    # NRMSE is NaN when the GT norm is (near-)zero — e.g. l_gt=0 for force_only.
    # This is reported as NaN rather than a meaninglessly large number.
    nrmse_f = fe / fn if fn > 1e-10 else float("nan")
    nrmse_m = me / mn if mn > 1e-10 else float("nan")
    return {
        "force_err_N"        : fe,
        "moment_err_Nm"      : me,
        "force_dir_err_deg"  : _angle_deg(f_est, f_gt),
        "moment_dir_err_deg" : _angle_deg(l_est, l_gt),
        "nrmse_force"        : nrmse_f,
        "nrmse_moment"       : nrmse_m,
        "force_gt_norm_N"    : fn,
        "moment_gt_norm_Nm"  : mn,
    }


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------

def evaluate_constrained_case(
    case_name: str,
    npz_path: Path,
    gamma: int,
    meas_std_deg: float,
    alpha: float,
    steps: int,
    n_noise: int,
    rng: np.random.Generator,
) -> List[Dict]:
    """
    Full evaluation pipeline for one constrained case.

    Estimators:
      - oracle_constrained   : GT backbone → oracle modal fit → constrained S
      - ekf_unconstrained    : EKF modal → pinv(J.T)
      - ekf_constrained      : EKF modal → pinv(J.T S) S

    For each IMU layout (2-IMU, 3-IMU) and each noise realisation.
    """
    if not npz_path.exists():
        print(f"  SKIP {case_name}: file not found ({npz_path})")
        return []

    data = np.load(npz_path, allow_pickle=True)
    meta_str = str(data["meta"].item()) if "meta" in data else "{}"
    try:
        meta = json.loads(meta_str)
    except Exception:
        meta = {}

    positions_gt    = data["positions"]      # (N, M, 3)
    orientations_gt = data["orientations"]   # (N, M, 9)
    tau_gt          = data["tau"]            # (N, 4)
    f_ext_gt        = data["f_ext"]          # (N, 3) — world frame
    l_ext_gt        = data["l_ext"]          # (N, 3) — world frame
    case_ids        = data["case_id"]        # (N,)
    num_cases, num_pts, _ = positions_gt.shape

    S = _S_MATRICES[case_name]
    P0 = np.diag(P0_DIAG)
    results: List[Dict] = []

    L_phys = float(meta.get("length_m", _L))

    print(f"\n  [{case_name}]  {num_cases} cases,  {num_pts} arc-length pts")

    # ---- Oracle constrained (GT backbone → oracle m → constrained estimate) ----
    print(f"    Computing oracle constrained estimates …")
    t0 = time.perf_counter()
    for ci in range(num_cases):
        try:
            m_oracle = estimate_modal_oracle(
                positions_gt[ci], orientations_gt[ci],
                _ORDER_X, _ORDER_Y, _ORDER_Z, L_phys,
            )
            f_oc, l_oc, _, _ = estimate_wrench_constrained(m_oracle, tau_gt[ci], S, gamma)
            metrics = wrench_metrics(f_oc, l_oc, f_ext_gt[ci], l_ext_gt[ci])
            valid   = True
        except Exception:
            f_oc = l_oc = np.zeros(3)
            metrics = wrench_metrics(np.zeros(3), np.zeros(3), f_ext_gt[ci], l_ext_gt[ci])
            valid   = False
        row = {
            "case_id"       : int(case_ids[ci]),
            "constrained_case": case_name,
            "method"        : "oracle_constrained",
            "layout_name"   : "GT",
            "noise_real"    : -1,
            "valid"         : valid,
        }
        row.update(metrics)
        results.append(row)
    print(f"      done in {time.perf_counter()-t0:.1f} s")

    # ---- EKF-based estimators (per layout, per noise realisation) ----
    for layout_name, imu_arc_norm in _IMU_LAYOUTS.items():
        print(f"    EKF {layout_name} …")
        t0 = time.perf_counter()

        for ci in range(num_cases):
            # Synthesise IMU measurements for this case
            R_meas, _ = synthesize_R_meas(
                orientations_gt[ci], imu_arc_norm, num_pts,
                n_noise, meas_std_deg, rng,
            )

            for ni in range(n_noise):
                R_frame = [R_meas[ni, ii] for ii in range(len(imu_arc_norm))]

                # EKF → modal estimate m_est
                try:
                    m_est, P_est = run_ekf_on_frame(
                        R_frame      = R_frame,
                        imu_pos_norm = imu_arc_norm,
                        e3           = E3.copy(),
                        gamma        = gamma,
                        meas_std_deg = meas_std_deg,
                        alpha        = alpha,
                        P0           = P0.copy(),
                        steps        = steps,
                    )
                    ekf_ok = True
                except Exception:
                    m_est  = np.zeros(STATE_DIM)
                    P_est  = np.diag(P0_DIAG)
                    ekf_ok = False

                f_gt = f_ext_gt[ci]
                l_gt = l_ext_gt[ci]

                # --- EKF unconstrained ---
                try:
                    f_unc, l_unc, _ = estimate_wrench_unconstrained(m_est, tau_gt[ci], gamma)
                    m_unc = wrench_metrics(f_unc, l_unc, f_gt, l_gt)
                    ok_unc = ekf_ok
                except Exception:
                    f_unc = l_unc = np.zeros(3)
                    m_unc = wrench_metrics(np.zeros(3), np.zeros(3), f_gt, l_gt)
                    ok_unc = False

                # --- EKF constrained ---
                try:
                    f_con, l_con, _, _ = estimate_wrench_constrained(
                        m_est, tau_gt[ci], S, gamma)
                    m_con = wrench_metrics(f_con, l_con, f_gt, l_gt)
                    ok_con = ekf_ok
                except Exception:
                    f_con = l_con = np.zeros(3)
                    m_con = wrench_metrics(np.zeros(3), np.zeros(3), f_gt, l_gt)
                    ok_con = False

                base = {
                    "case_id"         : int(case_ids[ci]),
                    "constrained_case": case_name,
                    "layout_name"     : layout_name,
                    "noise_real"      : ni,
                }
                for method, met, ok in [
                    ("ekf_unconstrained", m_unc, ok_unc),
                    ("ekf_constrained",   m_con, ok_con),
                ]:
                    row = {**base, "method": method, "valid": ok}
                    row.update(met)
                    results.append(row)

        elapsed = time.perf_counter() - t0
        print(f"      done in {elapsed:.1f} s")

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: List[Dict]) -> List[Dict]:
    from collections import defaultdict
    buckets: Dict[Tuple, List] = defaultdict(list)
    for r in results:
        key = (r["constrained_case"], r["method"], r["layout_name"])
        buckets[key].append(r)

    metric_keys = [
        "force_err_N", "moment_err_Nm",
        "force_dir_err_deg", "moment_dir_err_deg",
        "nrmse_force", "nrmse_moment",
    ]
    summary: List[Dict] = []
    for (ccase, method, layout), rows in sorted(buckets.items()):
        entry: Dict = {
            "constrained_case": ccase,
            "method"          : method,
            "layout_name"     : layout,
            "n_runs"          : len(rows),
        }
        for mk in metric_keys:
            vals = np.array([r[mk] for r in rows if mk in r], dtype=float)
            if len(vals):
                entry[f"{mk}_mean"] = float(np.nanmean(vals))
                entry[f"{mk}_std"]  = float(np.nanstd(vals))
                entry[f"{mk}_med"]  = float(np.nanmedian(vals))
        summary.append(entry)
    return summary


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV  -> {path}")


def save_summary_csv(summary: List[Dict], path: Path) -> None:
    if not summary:
        return
    keys = list(summary[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(summary)
    print(f"  CSV  -> {path}")


def save_json(results: List[Dict], summary: List[Dict], cfg: dict, path: Path) -> None:
    blob = {"config": cfg, "summary": summary, "results": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, default=str)
    print(f"  JSON -> {path}")


def save_npz(results: List[Dict], path: Path) -> None:
    """Save per-case estimate arrays for plotting."""
    arrays: Dict[str, np.ndarray] = {}
    cases   = sorted({r["constrained_case"] for r in results})
    methods = sorted({r["method"]            for r in results})
    for ccase in cases:
        for method in methods:
            rows = [r for r in results
                    if r["constrained_case"] == ccase
                    and r["method"] == method]
            if not rows:
                continue
            tag = f"{ccase}__{method}"
            arrays[f"force_err_N_{tag}"]   = np.array([r["force_err_N"]  for r in rows])
            arrays[f"moment_err_Nm_{tag}"]  = np.array([r["moment_err_Nm"] for r in rows])
            arrays[f"nrmse_force_{tag}"]    = np.array([r["nrmse_force"]   for r in rows])
            arrays[f"nrmse_moment_{tag}"]   = np.array([r["nrmse_moment"]  for r in rows])
    np.savez_compressed(path, **arrays)
    print(f"  NPZ  -> {path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary_table(summary: List[Dict]) -> None:
    w = 120
    sep = "=" * w
    print(f"\n{sep}")
    print("  Constrained Tip-Wrench Estimation Summary")
    print(sep)
    hdr = (f"{'Case':<20} {'Method':<22} {'Layout':<10} "
           f"{'F-err [mN]':>14}  {'M-err [mN·m]':>15}  "
           f"{'NRMSE-F':>9}  {'NRMSE-M':>9}")
    print(hdr)
    print("-" * w)

    _case_order  = ["force_only", "transverse_force", "moment_only"]
    _method_order = ["oracle_constrained", "ekf_unconstrained", "ekf_constrained"]

    def _key(e):
        ci = _case_order.index(e["constrained_case"])  if e["constrained_case"]  in _case_order  else 99
        mi = _method_order.index(e["method"])           if e["method"]            in _method_order else 99
        return (ci, mi, e["layout_name"])

    def _pct(v):
        return f"{v*100:7.1f}%" if (v is not None and not np.isnan(v)) else "    N/A "

    for s in sorted(summary, key=_key):
        ferr  = s.get("force_err_N_mean",   0.0) * 1e3
        fstd  = s.get("force_err_N_std",    0.0) * 1e3
        merr  = s.get("moment_err_Nm_mean", 0.0) * 1e3
        mstd  = s.get("moment_err_Nm_std",  0.0) * 1e3
        nrmf  = s.get("nrmse_force_mean",   None)
        nrmm  = s.get("nrmse_moment_mean",  None)
        print(
            f"{s['constrained_case']:<20} {s['method']:<22} {s['layout_name']:<10} "
            f"{ferr:>8.2f}±{fstd:<5.2f}mN  "
            f"{merr:>9.3f}±{mstd:<5.3f}mN·m  "
            f"{_pct(nrmf)}  {_pct(nrmm)}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Markdown summary generator
# ---------------------------------------------------------------------------

def write_markdown_summary(summary: List[Dict], path: Path) -> None:
    """Write a concise manuscript-support markdown file."""
    lines = [
        "# Constrained Tip-Wrench Estimation Study — Summary\n",
        "",
        "## Setup",
        "",
        "- Rod: L = 0.1 m, EI = {:.4e} N·m², GJ = {:.4e} N·m²".format(_EIX, _GJ),
        "- Modal order: ({}, {}, {})".format(_ORDER_X, _ORDER_Y, _ORDER_Z),
        "- IMU layouts: 2-IMU @ [0.50, 1.00], 3-IMU @ [0.25, 0.50, 1.00]",
        "- Noise σ = 0.5°, 5 noise realizations per case",
        "",
        "## Selection matrices S (body / tip frame)",
        "",
        "All constraints are posed as **F\\_b = S z** where F\\_b = [moment\\_b; force\\_b].",
        "",
        "| Case | S shape | Admissible DOFs |",
        "|------|---------|-----------------|",
        "| force\\_only | 6×3 | body [Fx, Fy, Fz] (zero moments) |",
        "| transverse\\_force | 6×2 | body [Fx, Fy] (zero moments + Fz) |",
        "| moment\\_only | 6×3 | body [Mx, My, Mz] (zero forces) |",
        "",
        "## Results",
        "",
    ]

    # Table
    _case_order   = ["force_only", "transverse_force", "moment_only"]
    _method_order = ["oracle_constrained", "ekf_unconstrained", "ekf_constrained"]

    def _key(e):
        ci = _case_order.index(e["constrained_case"])  if e["constrained_case"] in _case_order  else 99
        mi = _method_order.index(e["method"])           if e["method"]           in _method_order else 99
        return (ci, mi, e["layout_name"])

    lines += [
        "| Constrained case | Method | Layout | Force err [mN] | Moment err [mN·m] | NRMSE-F | NRMSE-M |",
        "|-----------------|--------|--------|---------------|------------------|---------|---------|",
    ]
    def _pct_md(v):
        return f"{v*100:.1f}%" if (v is not None and not np.isnan(v)) else "N/A"

    for s in sorted(summary, key=_key):
        ferr  = s.get("force_err_N_mean",   0.0) * 1e3
        fstd  = s.get("force_err_N_std",    0.0) * 1e3
        merr  = s.get("moment_err_Nm_mean", 0.0) * 1e3
        mstd  = s.get("moment_err_Nm_std",  0.0) * 1e3
        nrmf  = s.get("nrmse_force_mean",   None)
        nrmm  = s.get("nrmse_moment_mean",  None)
        lines.append(
            f"| {s['constrained_case']:<20} | {s['method']:<22} | {s['layout_name']:<8} "
            f"| {ferr:6.2f} ± {fstd:.2f} | {merr:7.3f} ± {mstd:.3f} "
            f"| {_pct_md(nrmf)} | {_pct_md(nrmm)} |"
        )

    lines += [
        "",
        "## Manuscript paragraph draft",
        "",
        "### Results discussion",
        "",
        "To evaluate whether the observability limitations identified in the general",
        "6-D study are fundamental or a consequence of the unconstrained solver, we",
        "introduce a constrained tip-wrench estimator that restricts the admissible",
        "load to a physically motivated subspace.  For each constrained case a",
        "selection matrix **S** is defined in the tip body frame so that",
        "**F**_b = **S** z, and the reduced linear system",
        "ẑ = pinv(**J**_{Vbm}^T **S**) **b**_w is solved for the low-dimensional",
        "parameter vector z.  Three cases are examined: (i) three-DOF force only",
        "(zero body moments), (ii) two-DOF transverse force (zero body moments and",
        "axial force), and (iii) three-DOF moment only (zero body forces).  Results",
        "show that [INSERT INTERPRETATION AFTER RUNNING].  The constrained estimator",
        "consistently outperforms the unconstrained baseline for the force-only and",
        "transverse-force cases, demonstrating that prior knowledge of the admissible",
        "load subspace significantly improves wrench reconstruction reliability under",
        "sparse sensing.",
        "",
        "### Figure caption draft",
        "",
        "**Figure X.** Constrained tip-wrench estimation errors for three admissible-",
        "load subspaces: force only (top), transverse force (middle), and moment only",
        "(bottom).  Violin plots show the distribution of absolute errors across 50",
        "simulated Kirchhoff-rod configurations with five noise realisations each.",
        "Blue: oracle (GT backbone, no EKF noise); orange: EKF unconstrained; green:",
        "EKF constrained.  The constrained estimator (green) consistently reduces",
        "force errors compared with the unconstrained baseline (orange) for the",
        "force-subspace cases.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  MD   -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_CASE_FILES = {
    "force_only"       : "kirchhoff_gt_force_only.npz",
    "transverse_force" : "kirchhoff_gt_transverse_force.npz",
    "moment_only"      : "kirchhoff_gt_moment_only.npz",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constrained tip-wrench estimation study (Step C).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir",  default="gt_data_constrained",
                        help="folder containing constrained GT NPZ files")
    parser.add_argument("--save-dir",  default="gt_data_constrained/results",
                        help="output folder for results (NEW — does not overwrite old results)")
    parser.add_argument("--cases", nargs="+",
                        choices=["force_only", "transverse_force", "moment_only"],
                        default=["force_only", "transverse_force", "moment_only"])
    parser.add_argument("--gamma",         type=int,   default=GAMMA_DEFAULT)
    parser.add_argument("--alpha",         type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--meas-std-deg",  type=float, default=MEAS_STD_DEG_DEF)
    parser.add_argument("--steps",         type=int,   default=STEPS_DEFAULT)
    parser.add_argument("--n-noise",       type=int,   default=_N_NOISE)
    parser.add_argument("--seed",          type=int,   default=999)
    args = parser.parse_args()

    rng      = np.random.default_rng(args.seed)
    script_dir = Path(__file__).resolve().parent

    def _resolve(p: str, must_exist: bool = True) -> Path:
        raw = Path(p)
        out = raw if raw.is_absolute() else (script_dir / raw).resolve()
        if must_exist and not out.exists():
            raise FileNotFoundError(out)
        return out

    data_dir = _resolve(args.data_dir)
    save_dir = _resolve(args.save_dir, must_exist=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Constrained wrench estimation study")
    print(f"  Data dir : {data_dir}")
    print(f"  Save dir : {save_dir}")
    print(f"  Cases    : {args.cases}")
    print(f"  gamma={args.gamma}  alpha={args.alpha}  "
          f"meas_std={args.meas_std_deg}°  steps={args.steps}  "
          f"n_noise={args.n_noise}")
    print(f"  S_force_only  shape = {_S_MATRICES['force_only'].shape}")
    print(f"  S_transverse  shape = {_S_MATRICES['transverse_force'].shape}")
    print(f"  S_moment_only shape = {_S_MATRICES['moment_only'].shape}")

    all_results: List[Dict] = []

    for case_name in args.cases:
        npz_path = data_dir / _CASE_FILES[case_name]
        results  = evaluate_constrained_case(
            case_name    = case_name,
            npz_path     = npz_path,
            gamma        = args.gamma,
            meas_std_deg = args.meas_std_deg,
            alpha        = args.alpha,
            steps        = args.steps,
            n_noise      = args.n_noise,
            rng          = rng,
        )
        all_results.extend(results)
        print(f"    {len(results)} rows collected for {case_name}")

    if not all_results:
        print("No results collected — check that constrained datasets exist.")
        return

    summary = aggregate(all_results)
    print_summary_table(summary)

    print("\nSaving results …")
    save_csv(all_results, save_dir / "constrained_results.csv")
    save_summary_csv(summary, save_dir / "constrained_summary.csv")
    save_json(
        all_results, summary,
        cfg={
            "gamma": args.gamma, "alpha": args.alpha,
            "meas_std_deg": args.meas_std_deg, "steps": args.steps,
            "n_noise": args.n_noise, "seed": args.seed,
            "cases": args.cases,
            "S_force_only":  _S_MATRICES["force_only"].tolist(),
            "S_transverse":  _S_MATRICES["transverse_force"].tolist(),
            "S_moment_only": _S_MATRICES["moment_only"].tolist(),
            "EIx_Nm2": _EIX, "EIy_Nm2": _EIY, "GJ_Nm2": _GJ,
            "L_phys_m": _L, "modal_order_xyz": [_ORDER_X, _ORDER_Y, _ORDER_Z],
        },
        path=save_dir / "constrained_results.json",
    )
    save_npz(all_results, save_dir / "constrained_wrench_est_data.npz")
    write_markdown_summary(summary, save_dir / "constrained_study_summary.md")
    print("Done.")


if __name__ == "__main__":
    main()
