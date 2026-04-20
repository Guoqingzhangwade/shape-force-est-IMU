#!/usr/bin/env python3
"""
Cross-Model Robustness Study — Step 3: EKF Evaluation Under Cross-Model Mismatch.

Loads Kirchhoff-rod ground-truth (Step 1) and synthetic IMU measurements (Step 2),
runs the finalized SO(3)-analytic EKF, and quantifies shape-estimation robustness
under cross-model mismatch.

Scientific role
---------------
This script implements:
    \\paragraph{Robustness Evaluation Under Cross-Model Mismatch.}

The estimator (modal curvature EKF) was tuned in Part A on a polynomial rod model.
Here it is evaluated on Kirchhoff/Cosserat-rod shapes without retuning.  Because
no ground-truth modal state exists for the Kirchhoff rod, ALL primary metrics are
geometry-based (position error, orientation error).

Estimator settings — fixed from matched-model study
----------------------------------------------------
  state         : m = [k0_x, k1_x, k0_y, k1_y, k0_z]
  Jacobian      : SO(3) analytic
  alpha         : 1.0
  P0            : diag([1.0, 0.5, 1.0, 0.5, 0.25])
  Q             : diag([PROC_STD0^2, PROC_STD1^2, ...])

Evaluation protocol
-------------------
For each (GT case i, noise realization r, sensor layout):
  1. Form a measurement frame from R_meas[i, r, :] (the saved noisy rotations).
  2. Run the EKF for `steps` iterations presenting this same frame at every step
     (stationary scene, repeated observation — identical to the matched-model protocol).
  3. Extract final m_est; reconstruct backbone geometry on the same arc-length grid
     as the Step-1 dataset.
  4. Compare estimated geometry against the Kirchhoff-rod ground truth.

Output files (in --save-dir)
-----------------------------
  kirchhoff_shape_est_results.csv          — per-(case, realization, layout)
  kirchhoff_shape_est_summary_by_layout.csv — aggregated per layout
  kirchhoff_shape_est_results.json         — full structured results
  kirchhoff_shape_est_summary.png/.pdf     — comparison bar plot
  kirchhoff_shape_est_overlays.png/.pdf    — representative shape overlays

Usage
-----
  python evaluate_kirchhoff_shape_estimation.py
  python evaluate_kirchhoff_shape_estimation.py --plot-summary
  python evaluate_kirchhoff_shape_estimation.py --plot-summary --plot-overlays
  python evaluate_kirchhoff_shape_estimation.py --steps 50 --save-dir results
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm

# ---------------------------------------------------------------------------
# Constants — fixed from matched-model study
# ---------------------------------------------------------------------------

STATE_DIM = 5
E3        = np.array([0.0, 0.0, 1.0])

P0_DIAG   = np.array([1.0, 0.5, 1.0, 0.5, 0.25])
PROC_STD0 = 1e-5    # process noise sigma for k0 terms
PROC_STD1 = 1e-6    # process noise sigma for k1 terms
Q_DIAG    = np.array([PROC_STD0**2, PROC_STD1**2,
                      PROC_STD0**2, PROC_STD1**2,
                      PROC_STD0**2])

ALPHA_DEFAULT    = 1.0
MEAS_STD_DEG_DEF = 0.5
STEPS_DEFAULT    = 30
GAMMA_DEFAULT    = 10

# 2-point Gauss-Legendre nodes on [0,1]
_GL_XI = np.array([0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6])


# ---------------------------------------------------------------------------
# SO(3) math helpers  (identical to sensor_placement_study_final.py)
# ---------------------------------------------------------------------------

def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2],  v[1]],
                     [v[2],  0.0, -v[0]],
                     [-v[1], v[0],  0.0]])


def vee(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def so3_log(R: np.ndarray) -> np.ndarray:
    val   = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(val)
    if abs(theta) < 1e-8:
        return vee(R - R.T) / 2.0
    return (theta / (2.0 * np.sin(theta))) * vee(R - R.T)


def jr_inv_so3(r: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(r)
    rhat  = skew(r)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * rhat
    c = 1.0 / theta**2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) + 0.5 * rhat + c * (rhat @ rhat)


# ---------------------------------------------------------------------------
# Magnus-expansion forward kinematics  (identical to sensor_placement_study_final.py)
# ---------------------------------------------------------------------------

def phi_mat(s: float) -> np.ndarray:
    """Shape-function matrix Phi(s): kappa(s) = Phi(s) @ m."""
    return np.array([[1, s, 0, 0, 0],
                     [0, 0, 1, s, 0],
                     [0, 0, 0, 0, 1]], dtype=float)


def twist_mat(k: np.ndarray, e3: np.ndarray) -> np.ndarray:
    return np.block([[skew(k), e3[:, None]],
                     [np.zeros((1, 3)), 0.0]])


def magnus_psi(s_i: float, h: float,
               m: np.ndarray, e3: np.ndarray) -> np.ndarray:
    c1 = s_i - h + _GL_XI[0] * h
    c2 = s_i - h + _GL_XI[1] * h
    e1 = twist_mat(phi_mat(c1) @ m, e3)
    e2 = twist_mat(phi_mat(c2) @ m, e3)
    return (h / 2) * (e1 + e2) + (np.sqrt(3) / 12) * h**2 * (e1 @ e2 - e2 @ e1)


def fwd_transform(m: np.ndarray, s: float,
                  e3: np.ndarray, gamma: int) -> np.ndarray:
    """Full 4×4 SE(3) transform at normalised arc-length s in [0, 1]."""
    T = np.eye(4)
    h = s / gamma
    for k in range(1, gamma + 1):
        T = T @ expm(magnus_psi(k * h, h, m, e3))
    return T


def fwd_rotation(m: np.ndarray, s: float,
                 e3: np.ndarray, gamma: int) -> np.ndarray:
    return fwd_transform(m, s, e3, gamma)[:3, :3]


def compute_dR_dm(m: np.ndarray, s: float,
                  gamma: int, e3: np.ndarray) -> np.ndarray:
    """Analytic dR/dm  shape (3, 3, STATE_DIM)."""
    Nm = len(m)
    h  = s / gamma

    Psi_list  = []
    ePsi_list = []
    T_before  = [np.eye(4)]
    dPsi_dm   = [np.zeros((4, 4, Nm)) for _ in range(gamma)]
    dePsi_dm  = [np.zeros((4, 4, Nm)) for _ in range(gamma)]

    for k in range(1, gamma + 1):
        Psi_k  = magnus_psi(k * h, h, m, e3)
        ePsi_k = expm(Psi_k)
        Psi_list.append(Psi_k)
        ePsi_list.append(ePsi_k)
        T_before.append(T_before[-1] @ ePsi_k)

    T_after_arr = [np.eye(4) for _ in range(gamma + 1)]
    for k in range(gamma - 1, 0, -1):
        T_after_arr[k] = ePsi_list[k] @ T_after_arr[k + 1]

    for k in range(1, gamma + 1):
        s_i        = k * h
        c1         = s_i - h + _GL_XI[0] * h
        c2         = s_i - h + _GL_XI[1] * h
        ph1, ph2   = phi_mat(c1), phi_mat(c2)
        eta1       = twist_mat(ph1 @ m, e3)
        eta2       = twist_mat(ph2 @ m, e3)
        for i in range(Nm):
            de1  = twist_mat(ph1[:, i], np.zeros(3))
            de2  = twist_mat(ph2[:, i], np.zeros(3))
            comm = (de1 @ eta2 - eta2 @ de1) + (eta1 @ de2 - de2 @ eta1)
            dPsi_dm[k-1][:, :, i] = (
                (h / 2) * (de1 + de2)
                + (np.sqrt(3) / 12) * h**2 * comm
            )

    for k in range(gamma):
        ePsi_k = ePsi_list[k]
        Psi_k  = Psi_list[k]
        for i in range(Nm):
            dP = dPsi_dm[k][:, :, i]
            dePsi_dm[k][:, :, i] = ePsi_k @ (dP + 0.5 * (Psi_k @ dP - dP @ Psi_k))

    dT_dm = np.zeros((4, 4, Nm))
    for i in range(Nm):
        for k in range(gamma):
            dT_dm[:, :, i] += T_before[k] @ dePsi_dm[k][:, :, i] @ T_after_arr[k + 1]
    return dT_dm[:3, :3, :]


# ---------------------------------------------------------------------------
# EKF measurement model
# ---------------------------------------------------------------------------

def so3_analytic_H_and_r(
    m: np.ndarray,
    imu_pos: np.ndarray,
    meas_frame: List[np.ndarray],
    e3: np.ndarray,
    gamma: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SO(3) log-residual and analytic Jacobian for all sensors.

    r_i = log(R_meas_i @ R_pred_i^T)
    H_i = -Jr^{-1}(r_i) @ vee(skew(dR/dm @ R_pred^T))
    """
    H_rows, r_rows = [], []
    for s_i, R_meas in zip(imu_pos, meas_frame):
        R_pred = fwd_rotation(m, float(s_i), e3, gamma)
        R_e    = R_meas @ R_pred.T
        r      = so3_log(R_e)
        Jr_inv = jr_inv_so3(r)
        dR_dm  = compute_dR_dm(m, float(s_i), gamma, e3)
        H_i    = np.zeros((3, STATE_DIM))
        for p in range(STATE_DIM):
            A = dR_dm[:, :, p] @ R_pred.T
            H_i[:, p] = -Jr_inv @ vee(0.5 * (A - A.T))
        H_rows.append(H_i)
        r_rows.append(r)
    return np.vstack(H_rows), np.hstack(r_rows)


# ---------------------------------------------------------------------------
# Shape reconstruction from modal state
# ---------------------------------------------------------------------------

def reconstruct_shape(
    m_est: np.ndarray,
    s_grid: np.ndarray,
    e3: np.ndarray,
    gamma: int,
    L_phys: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover backbone positions and orientations from modal state m.

    The SE(3) Magnus product T(s)[:3,3] gives position in normalised units
    (s in [0,1]); multiply by L_phys to obtain physical metres.

    Returns
    -------
    positions    : (M, 3)    in metres
    orientations : (M, 3, 3) rotation matrices R(s)
    """
    M = len(s_grid)
    positions    = np.zeros((M, 3))
    orientations = np.zeros((M, 3, 3))
    for i, s in enumerate(s_grid):
        T             = fwd_transform(m_est, float(s), e3, gamma)
        positions[i]  = T[:3, 3] * L_phys
        orientations[i] = T[:3, :3]
    return positions, orientations


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------

def compute_geometry_metrics(
    p_est: np.ndarray,
    R_est: np.ndarray,
    p_gt: np.ndarray,
    R_gt: np.ndarray,
) -> Dict[str, float]:
    """
    Geometry-based shape-estimation errors.

    Parameters
    ----------
    p_est, p_gt : (M, 3)    positions in metres
    R_est, R_gt : (M, 3, 3) rotation matrices

    Returns keys (all floats)
    -------------------------
    mean_centerline_error      [m]
    max_centerline_error       [m]
    tip_position_error         [m]
    tip_orientation_error_deg  [deg]
    mean_orientation_error_deg [deg]
    """
    pos_err = np.linalg.norm(p_est - p_gt, axis=1)   # (M,)

    ori_err = np.empty(len(R_est))
    for i in range(len(R_est)):
        ori_err[i] = np.linalg.norm(so3_log(R_est[i] @ R_gt[i].T))

    return {
        "mean_centerline_error"    : float(np.mean(pos_err)),
        "max_centerline_error"     : float(np.max(pos_err)),
        "tip_position_error"       : float(pos_err[-1]),
        "tip_orientation_error_deg": float(np.rad2deg(ori_err[-1])),
        "mean_orientation_error_deg": float(np.rad2deg(np.mean(ori_err))),
    }


# ---------------------------------------------------------------------------
# EKF runner
# ---------------------------------------------------------------------------

def run_ekf_on_frame(
    R_frame: List[np.ndarray],
    imu_pos_norm: np.ndarray,
    e3: np.ndarray,
    gamma: int,
    meas_std_deg: float,
    alpha: float,
    P0: np.ndarray,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run SO(3)-analytic EKF for `steps` iterations.

    The same measurement frame is presented at every step (stationary scene,
    repeated observation).  This matches the matched-model protocol and lets
    the filter converge to the best-fit modal state for this observation.

    Parameters
    ----------
    R_frame      : list of (3,3) arrays — one per IMU sensor
    imu_pos_norm : (n_imu,) normalised arc-length positions in [0, 1]

    Returns
    -------
    m_est : (STATE_DIM,) final modal estimate
    P_est : (STATE_DIM, STATE_DIM) final covariance
    """
    sigma  = np.deg2rad(meas_std_deg)
    R_sngl = alpha * sigma**2 * np.eye(3)
    Q      = np.diag(Q_DIAG)
    n_imu  = len(imu_pos_norm)
    R_big  = np.kron(np.eye(n_imu), R_sngl)

    m_est = np.zeros(STATE_DIM)
    P_est = P0.copy()

    for _ in range(steps):
        P_pred = P_est + Q
        H, r   = so3_analytic_H_and_r(m_est, imu_pos_norm, R_frame, e3, gamma)
        innov  = -r
        S      = H @ P_pred @ H.T + R_big
        K      = P_pred @ H.T @ inv(S)
        m_est  = m_est + K @ innov
        IKH    = np.eye(STATE_DIM) - K @ H
        P_est  = IKH @ P_pred @ IKH.T + K @ R_big @ K.T
        P_est  = 0.5 * (P_est + P_est.T)

    return m_est, P_est


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth_dataset(
    npz_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Returns positions (N,M,3), orientations (N,M,9), case_ids (N,), meta."""
    data         = np.load(npz_path, allow_pickle=True)
    positions    = data["positions"]
    orientations = data["orientations"]
    case_ids     = data["case_id"]
    meta: dict   = {}
    if "meta" in data:
        try:
            meta = json.loads(str(data["meta"].item()))
        except Exception:
            pass
    json_path = npz_path.with_suffix(".json")
    if json_path.exists():
        try:
            meta.update(json.loads(json_path.read_text()))
        except Exception:
            pass
    return positions, orientations, case_ids, meta


def load_imu_measurements(
    npz_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Returns R_true, R_meas, imu_positions, imu_arc_indices, meta."""
    data            = np.load(npz_path, allow_pickle=True)
    R_true          = data["R_true"]
    R_meas          = data["R_meas"]
    imu_positions   = data["imu_positions"]
    imu_arc_indices = data["imu_arc_indices"]
    meta: dict      = {}
    if "meta" in data:
        try:
            meta = json.loads(str(data["meta"].item()))
        except Exception:
            pass
    return R_true, R_meas, imu_positions, imu_arc_indices, meta


# ---------------------------------------------------------------------------
# Per-layout evaluation
# ---------------------------------------------------------------------------

def evaluate_layout(
    positions_gt: np.ndarray,
    orientations_gt: np.ndarray,
    case_ids: np.ndarray,
    R_meas: np.ndarray,
    imu_pos_norm: np.ndarray,
    s_grid: np.ndarray,
    L_phys: float,
    e3: np.ndarray,
    gamma: int,
    meas_std_deg: float,
    alpha: float,
    P0: np.ndarray,
    steps: int,
    layout_name: str,
) -> List[Dict]:
    """
    Run EKF for all cases × noise realizations in one sensor layout.
    Returns list of result dicts; private keys (prefixed '_') hold arrays
    needed for plotting and are stripped before JSON/CSV export.
    """
    N, n_noise, n_imu, _, _ = R_meas.shape
    M_gt   = positions_gt.shape[1]
    R_gt_all = orientations_gt.reshape(N, M_gt, 3, 3)

    results: List[Dict] = []
    t0 = time.perf_counter()

    for i in range(N):
        p_gt = positions_gt[i]   # (M_gt, 3)
        R_gt = R_gt_all[i]       # (M_gt, 3, 3)

        for r in range(n_noise):
            R_frame = [R_meas[i, r, j] for j in range(n_imu)]

            m_est, _ = run_ekf_on_frame(
                R_frame      = R_frame,
                imu_pos_norm = imu_pos_norm,
                e3           = e3,
                gamma        = gamma,
                meas_std_deg = meas_std_deg,
                alpha        = alpha,
                P0           = P0,
                steps        = steps,
            )

            p_est, R_est = reconstruct_shape(m_est, s_grid, e3, gamma, L_phys)
            metrics      = compute_geometry_metrics(p_est, R_est, p_gt, R_gt)

            results.append({
                "case_id"           : int(case_ids[i]),
                "layout_name"       : layout_name,
                "num_imus"          : n_imu,
                "noise_realization" : r,
                **metrics,
                "_m_est" : m_est,
                "_p_est" : p_est,
                "_R_est" : R_est,
            })

        if (i + 1) % 10 == 0 or (i + 1) == N:
            elapsed = time.perf_counter() - t0
            print(f"    [{layout_name}] case {i+1:3d}/{N}  "
                  f"elapsed {elapsed:.1f} s")

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "mean_centerline_error",
    "max_centerline_error",
    "tip_position_error",
    "tip_orientation_error_deg",
    "mean_orientation_error_deg",
]


def aggregate_by_layout(results: List[Dict]) -> Dict[str, Dict]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        groups[r["layout_name"]].append(r)
    summary: Dict[str, Dict] = {}
    for name, rows in groups.items():
        s: Dict = {"n": len(rows), "num_imus": rows[0]["num_imus"]}
        for k in METRIC_KEYS:
            vals = np.array([row[k] for row in rows])
            s[k + "_mean"] = float(np.mean(vals))
            s[k + "_std"]  = float(np.std(vals))
            s[k + "_med"]  = float(np.median(vals))
        summary[name] = s
    return summary


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

PERCASE_FIELDS = [
    "case_id", "layout_name", "num_imus", "noise_realization",
    "mean_centerline_error", "max_centerline_error",
    "tip_position_error", "tip_orientation_error_deg",
    "mean_orientation_error_deg",
]


def save_results_csv(results: List[Dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PERCASE_FIELDS)
        w.writeheader()
        for row in results:
            w.writerow({k: row[k] for k in PERCASE_FIELDS})
    print(f"  Saved → {path}")


def save_summary_csv(summary: Dict[str, Dict], path: Path) -> None:
    fields = ["layout_name", "num_imus", "n"] + [
        f"{k}_{s}" for k in METRIC_KEYS for s in ("mean", "std", "med")
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, s in summary.items():
            w.writerow({"layout_name": name, **s})
    print(f"  Saved → {path}")


def save_results_json(
    results: List[Dict],
    summary: Dict[str, Dict],
    cfg: Dict,
    path: Path,
) -> None:
    clean = [{k: v for k, v in r.items() if not k.startswith("_")}
             for r in results]
    path.write_text(json.dumps({
        "config"  : cfg,
        "summary" : summary,
        "per_case": clean,
        "date"    : datetime.datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    print(f"  Saved → {path}")


def save_shapes_npz(
    all_results: List[Dict],
    layout_names: List[str],
    path: Path,
) -> None:
    """
    Save estimated backbone positions for all layouts so that the
    plot-only companion (plot_kirchhoff_shape_est.py) can regenerate
    shape overlays without re-running the EKF.

    Per-layout arrays in the NPZ
    ----------------------------
    p_est_{name}     : (n_rows, M, 3)  estimated positions [m]
    case_ids_{name}  : (n_rows,)       case_id per row
    noise_real_{name}: (n_rows,)       noise_realization index per row
    layout_names     : object array of layout name strings
    """
    arrays: Dict = {"layout_names": np.array(layout_names, dtype=object)}
    for lname in layout_names:
        rows = [r for r in all_results if r["layout_name"] == lname]
        arrays[f"p_est_{lname}"]     = np.stack([r["_p_est"] for r in rows])
        arrays[f"case_ids_{lname}"]  = np.array([r["case_id"] for r in rows])
        arrays[f"noise_real_{lname}"] = np.array([r["noise_realization"] for r in rows])
    np.savez(path, **arrays)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_cross_model_summary(
    summary: Dict[str, Dict],
    save_stem: Path | None = None,
) -> None:
    """
    Two-panel bar plot: mean centerline error and tip position error
    for each sensor layout.  Error bars = ±1 std.  Values annotated in mm.
    """
    import matplotlib.pyplot as plt

    names     = list(summary.keys())
    n         = len(names)
    x         = np.arange(n)
    colors    = plt.cm.tab10(np.linspace(0, 0.5, n))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    panels = [
        ("mean_centerline_error", "Mean centerline error  [m]"),
        ("tip_position_error",    "Tip position error  [m]"),
    ]
    for ax, (metric, ylabel) in zip(axes, panels):
        means = [summary[nm][metric + "_mean"] for nm in names]
        stds  = [summary[nm][metric + "_std"]  for nm in names]
        bars  = ax.bar(x, means, yerr=stds, color=colors, capsize=5,
                       width=0.5, alpha=0.85, error_kw=dict(lw=1.2))
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=12, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{m*1e3:.2f} mm", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Cross-Model Robustness: Kirchhoff-rod shape estimation",
                 fontsize=11)
    plt.tight_layout()
    plt.show()
    if save_stem is not None:
        for ext in ("png", "pdf"):
            p = save_stem.parent / (save_stem.name + f"_summary.{ext}")
            fig.savefig(p, dpi=150)
            print(f"  Figure saved → {p}")
    plt.close(fig)


def plot_representative_overlays(
    positions_gt: np.ndarray,
    case_ids: np.ndarray,
    all_results: List[Dict],
    layout_names: List[str],
    n_cases: int = 3,
    save_stem: Path | None = None,
) -> None:
    """
    Shape overlay for n_cases representative cases (low / median / high error).

    For each case shows:
      - Kirchhoff-rod GT backbone  (black solid)
      - EKF-estimated backbone per layout  (coloured dashed)
    First noise realization is used for the overlay.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Build case-id → array-index map
    cid_to_idx = {int(cid): idx for idx, cid in enumerate(case_ids)}

    # Select representative cases ranked by mean-centerline-error
    # of the first layout, noise-averaged.
    layout0 = layout_names[0]
    case_errs: Dict[int, List[float]] = defaultdict(list)
    for r in all_results:
        if r["layout_name"] == layout0:
            case_errs[r["case_id"]].append(r["mean_centerline_error"])
    case_mean = {cid: float(np.mean(v)) for cid, v in case_errs.items()}

    sorted_cases = sorted(case_mean.items(), key=lambda x: x[1])
    n_total  = len(sorted_cases)
    pick_idx = [0, n_total // 2, n_total - 1][:n_cases]
    rep_cases = [sorted_cases[i][0] for i in pick_idx]
    labels    = ["low error", "median error", "high error"][:n_cases]

    layout_colors = plt.cm.Set1(np.linspace(0, 0.55, len(layout_names)))

    fig = plt.figure(figsize=(5 * n_cases, 5))
    for col, (cid, lbl) in enumerate(zip(rep_cases, labels)):
        ax  = fig.add_subplot(1, n_cases, col + 1, projection="3d")
        idx = cid_to_idx[cid]
        p_gt = positions_gt[idx]

        ax.plot(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2],
                "k-", lw=1.8, label="GT (Kirchhoff)")

        for li, lname in enumerate(layout_names):
            rows = [r for r in all_results
                    if r["case_id"] == cid
                    and r["layout_name"] == lname
                    and r["noise_realization"] == 0]
            if not rows:
                continue
            p_est = rows[0]["_p_est"]
            ax.plot(p_est[:, 0], p_est[:, 1], p_est[:, 2],
                    "--", color=layout_colors[li], lw=1.3,
                    label=lname, alpha=0.9)

        ax.set_title(f"case {cid}  ({lbl})", fontsize=8)
        ax.set_xlabel("x [m]", fontsize=7)
        ax.set_ylabel("y [m]", fontsize=7)
        ax.set_zlabel("z [m]", fontsize=7)
        ax.set_box_aspect([1, 1, 1])
        if col == 0:
            ax.legend(fontsize=6, loc="upper left")

    fig.suptitle("Shape overlays — GT vs EKF estimate (cross-model)", fontsize=10)
    plt.tight_layout()
    plt.show()
    if save_stem is not None:
        for ext in ("png", "pdf"):
            p = save_stem.parent / (save_stem.name + f"_overlays.{ext}")
            fig.savefig(p, dpi=150)
            print(f"  Figure saved → {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: EKF robustness under cross-model mismatch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt",   default="gt_data/kirchhoff_gt_dataset.npz",
                        help="Step-1 ground-truth NPZ")
    parser.add_argument("--imu2", default="gt_data/kirchhoff_imu_2imu.npz",
                        help="Step-2 2-IMU measurement NPZ")
    parser.add_argument("--imu3", default="gt_data/kirchhoff_imu_3imu.npz",
                        help="Step-2 3-IMU measurement NPZ")
    parser.add_argument("--save-dir",  default="gt_data/results")
    parser.add_argument("--steps",     type=int,   default=STEPS_DEFAULT,
                        help="EKF iterations per (case, realization) run")
    parser.add_argument("--gamma",     type=int,   default=GAMMA_DEFAULT)
    parser.add_argument("--alpha",     type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--meas-std-deg", type=float, default=MEAS_STD_DEG_DEF)
    parser.add_argument("--plot-summary",  action="store_true",
                        help="show and save aggregate comparison plot")
    parser.add_argument("--plot-overlays", action="store_true",
                        help="show and save representative shape overlays")
    parser.add_argument("--num-overlay-cases", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading ground-truth dataset …")
    positions_gt, orientations_gt, case_ids, gt_meta = \
        load_ground_truth_dataset(Path(args.gt))
    N, num_pts, _ = positions_gt.shape
    L_phys = float(gt_meta.get("length_m", 0.10))
    s_grid = np.linspace(0.0, 1.0, num_pts)
    print(f"  {N} cases,  {num_pts} arc-length points,  L = {L_phys} m")

    print("\nLoading IMU measurement files …")
    imu_datasets: Dict[str, Dict] = {}
    for label, path_str in [("2-IMU", args.imu2), ("3-IMU", args.imu3)]:
        p = Path(path_str)
        if not p.exists():
            print(f"  WARNING: {p} not found — skipping {label}")
            continue
        R_true, R_meas, imu_pos, imu_idx, imu_meta = load_imu_measurements(p)
        # Use exact arc-length values saved in Step-2 metadata
        imu_actual_s = np.array(
            imu_meta.get("imu_actual_s", (imu_idx / (num_pts - 1)).tolist()),
            dtype=float,
        )
        imu_datasets[label] = {
            "R_meas" : R_meas,
            "imu_pos": imu_actual_s,
        }
        pos_str = "{" + ", ".join(f"{v:.4f}" for v in imu_actual_s) + "}"
        print(f"  {label}: R_meas {R_meas.shape},  imu_pos = {pos_str}")

    if not imu_datasets:
        raise RuntimeError("No IMU measurement files found. Run Step 2 first.")

    P0 = np.diag(P0_DIAG)

    # ------------------------------------------------------------------
    # Evaluate each layout
    # ------------------------------------------------------------------
    all_results: List[Dict] = []

    for layout_name, ds in imu_datasets.items():
        print(f"\nEvaluating {layout_name} …")
        layout_results = evaluate_layout(
            positions_gt    = positions_gt,
            orientations_gt = orientations_gt,
            case_ids        = case_ids,
            R_meas          = ds["R_meas"],
            imu_pos_norm    = ds["imu_pos"],
            s_grid          = s_grid,
            L_phys          = L_phys,
            e3              = E3.copy(),
            gamma           = args.gamma,
            meas_std_deg    = args.meas_std_deg,
            alpha           = args.alpha,
            P0              = P0,
            steps           = args.steps,
            layout_name     = layout_name,
        )
        all_results.extend(layout_results)
        print(f"  {len(layout_results)} runs complete.")

    # ------------------------------------------------------------------
    # Aggregate and print
    # ------------------------------------------------------------------
    summary = aggregate_by_layout(all_results)

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Cross-Model Robustness  "
          f"(alpha={args.alpha}, steps={args.steps}, gamma={args.gamma})")
    print(sep)
    print(f"{'Layout':<10}  {'n_IMU':>5}  "
          f"{'mean_pos [mm]':>14}  {'tip_pos [mm]':>13}  "
          f"{'tip_ori [deg]':>14}")
    print("-" * 72)
    for name, s in summary.items():
        print(
            f"{name:<10}  {s['num_imus']:>5}  "
            f"{s['mean_centerline_error_mean']*1e3:>10.3f}"
            f" ±{s['mean_centerline_error_std']*1e3:<6.3f}"
            f"{s['tip_position_error_mean']*1e3:>10.3f}"
            f" ±{s['tip_position_error_std']*1e3:<6.3f}"
            f"{s['tip_orientation_error_deg_mean']:>10.3f}"
            f" ±{s['tip_orientation_error_deg_std']:.3f}"
        )
    print(sep)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    stem = save_dir / "kirchhoff_shape_est"
    save_results_csv(all_results, stem.parent / (stem.name + "_results.csv"))
    save_summary_csv(summary,    stem.parent / (stem.name + "_summary_by_layout.csv"))
    save_results_json(
        all_results, summary,
        cfg={
            "gt": args.gt, "imu2": args.imu2, "imu3": args.imu3,
            "steps": args.steps, "gamma": args.gamma,
            "alpha": args.alpha, "meas_std_deg": args.meas_std_deg,
            "P0_diag": P0_DIAG.tolist(), "Q_diag": Q_DIAG.tolist(),
            "L_phys": L_phys, "num_pts": num_pts,
        },
        path=stem.parent / (stem.name + "_results.json"),
    )
    save_shapes_npz(all_results, layout_names,
                    stem.parent / (stem.name + "_shapes.npz"))

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    layout_names = list(imu_datasets.keys())

    if args.plot_summary:
        plot_cross_model_summary(summary, save_stem=stem)

    if args.plot_overlays:
        plot_representative_overlays(
            positions_gt = positions_gt,
            case_ids     = case_ids,
            all_results  = all_results,
            layout_names = layout_names,
            n_cases      = args.num_overlay_cases,
            save_stem    = stem,
        )


if __name__ == "__main__":
    main()
