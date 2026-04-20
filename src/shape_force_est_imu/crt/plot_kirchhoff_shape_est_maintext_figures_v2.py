#!/usr/bin/env python3
"""
plot_kirchhoff_shape_est_maintext_figures_v2.py
===============================================
New plot-only script for the Cross-Model Robustness Study.
Generates TWO publication-quality figures without re-running the EKF.

Figure 1 — Main aggregate (2 × 2, all 50 cases × 5 realisations)
-----------------------------------------------------------------
(a) Grouped violin: mean centerline error + tip position error  [mm]
(b) Grouped violin: mean orientation error + tip orientation error  [deg]
(c) Global arc-length position error  mean + IQR  [mm]
(d) Global arc-length SO(3) orientation error  mean + IQR  [deg]

Figure 2 — Representative-case detail  (n_cases cols × 3 rows)
--------------------------------------------------------------
Row 0: Robot-style 3D overlay per case
        · GT shown as backbone + semi-transparent disks + sparse RGB frames
        · 2-IMU estimate: blue dashed centerline
        · 3-IMU estimate: red dash-dot centerline
Row 1: Per-case arc-length position error  mean ± IQR  [mm]
Row 2: Per-case arc-length SO(3) orientation error  mean ± IQR  [deg]

Cases selected from the top-60 % most nonplanar shapes, spread across
complexity bands (low / mid-low / mid-high / high by default).

Style reference
---------------
3D rendering style (disks, axis-off clean panels) is adapted from:
  scripts/archive/crt/3d_CR_with_disc_tendon_v2.py
  Specifically: plot_cylinder(), the axis-off / pane-hide block, and
  set_equal_axis_scale().  The heavy conical-arrow frame renderer from
  that script was NOT used; lightweight quiver arrows are used instead
  to keep multi-panel rendering fast.

Disk / frame parameters
-----------------------
  GT disks   : 6 locations, s = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
               radius = 0.0055 m (~5.5 % of arc length), alpha = 0.18
  GT frames  : 5 locations, s = 0.2, 0.4, 0.6, 0.8, 1.0
               scale = 0.007 m, RGB quiver arrows
  3-IMU frames: same 5 locations, lighter alpha = 0.30, small quivers
  2-IMU frames: not shown (avoids clutter)

Required input files
--------------------
  gt_data/kirchhoff_gt_dataset.npz
  gt_data/results/kirchhoff_shape_est_results.json
  gt_data/results/kirchhoff_shape_est_shapes.npz

Output files
------------
  gt_data/results/kirchhoff_shape_est_main_aggregate_violin_v2.png
  gt_data/results/kirchhoff_shape_est_main_aggregate_violin_v2.pdf
  gt_data/results/kirchhoff_shape_est_composite_maintext_v2.png
  gt_data/results/kirchhoff_shape_est_composite_maintext_v2.pdf

Usage
-----
  python plot_kirchhoff_shape_est_maintext_figures_v2.py          # both
  python plot_kirchhoff_shape_est_maintext_figures_v2.py --plot-aggregate-only
  python plot_kirchhoff_shape_est_maintext_figures_v2.py --plot-representative-only
  python plot_kirchhoff_shape_est_maintext_figures_v2.py --no-save
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------------------------------------------------------------------
# Style  (slightly larger than composite backup script)
# ---------------------------------------------------------------------------

LABEL_FS  = 13
TICK_FS   = 12
TITLE_FS  = 13
PANEL_FS  = 14
LEGEND_FS = 12
ANN_FS    = 9.0

COLORS = {"2-IMU": "#1f77b4", "3-IMU": "#d62728"}
LS     = {"2-IMU": "--",      "3-IMU": "-."}
LW     = 2.0

_GROUP_CENTRES = [1.0, 3.0]
_HALF_GAP      = 0.22
_VWIDTH        = 0.36

_IMU_S = {"2-IMU": [0.50, 1.00], "3-IMU": [0.25, 0.50, 1.00]}

_ELEV, _AZIM = 25.0, -55.0

# 3-D disk / frame geometry (metres — rod arc-length ≈ 0.10 m)
_DISK_RADIUS     = 0.0055   # m
_DISK_HEIGHT     = 0.0010   # m
_DISK_COLOR      = "lightgrey"
_DISK_ALPHA      = 0.18
_DISK_EDGE_ALPHA = 0.45
_DISK_S_LOCS     = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)   # 6 disks
_FRAME_SCALE     = 0.007    # m
_FRAME_LW        = 1.2
_FRAME_S_LOCS    = (0.2, 0.4, 0.6, 0.8, 1.0)          # 5 frames
_FRAME_COLS      = ("r", "g", "b")                     # x y z axes
_FRAME_ALPHA_GT  = 0.90
_FRAME_ALPHA_EST = 0.30    # lighter frames for 3-IMU

# complexity-band labels for n_cases = 4
_CX_LABELS = {
    2: ("low complexity",  "high complexity"),
    3: ("low complexity",  "mid complexity",     "high complexity"),
    4: ("low complexity",  "mid-low complexity", "mid-high complexity", "high complexity"),
    5: ("complexity 1/5",  "complexity 2/5",     "complexity 3/5",
        "complexity 4/5",  "complexity 5/5"),
}


# ---------------------------------------------------------------------------
# SO(3) helper
# ---------------------------------------------------------------------------

def _vee(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def _so3_log(R: np.ndarray) -> np.ndarray:
    val   = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(val)
    if abs(theta) < 1e-8:
        return _vee(R - R.T) / 2.0
    return (theta / (2.0 * np.sin(theta))) * _vee(R - R.T)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_saved_results(
    results_dir: Path,
    gt_npz: Path,
) -> dict:
    """Load all saved results and GT data; return a single bundle dict."""
    json_path   = results_dir / "kirchhoff_shape_est_results.json"
    shapes_path = results_dir / "kirchhoff_shape_est_shapes.npz"
    for p in (json_path, shapes_path, gt_npz):
        if not p.exists():
            raise FileNotFoundError(p)

    raw = json.loads(json_path.read_text())
    per_case     = raw["per_case"]
    layout_names = list(raw["summary"].keys())

    gt_data       = np.load(gt_npz, allow_pickle=True)
    N, M, _       = gt_data["positions"].shape
    positions_gt  = gt_data["positions"]
    orientations_gt = gt_data["orientations"].reshape(N, M, 3, 3)
    case_ids_gt   = gt_data["case_id"]

    shapes_data = np.load(shapes_path, allow_pickle=True)
    shapes: Dict[str, dict] = {}
    for nm in layout_names:
        entry: dict = {
            "p_est"     : shapes_data[f"p_est_{nm}"],
            "case_ids"  : shapes_data[f"case_ids_{nm}"],
            "noise_real": shapes_data[f"noise_real_{nm}"],
        }
        if f"R_est_{nm}" in shapes_data.files:
            entry["R_est"] = shapes_data[f"R_est_{nm}"]
        shapes[nm] = entry

    n_cases = len(positions_gt)
    n_noise = len(shapes[layout_names[0]]["p_est"]) // n_cases
    s_grid  = np.linspace(0.0, 1.0, M)

    print(f"JSON  : {len(per_case)} rows, layouts {layout_names}")
    print(f"GT    : {n_cases} cases, {M} arc-length points")
    print(f"Shapes: n_noise~{n_noise}, R_est={'R_est' in shapes[layout_names[0]]}")

    return dict(
        per_case=per_case,
        layout_names=layout_names,
        positions_gt=positions_gt,
        orientations_gt=orientations_gt,
        case_ids_gt=case_ids_gt,
        shapes=shapes,
        n_cases=n_cases,
        n_noise=n_noise,
        M=M,
        s_grid=s_grid,
    )


def _cid_map(case_ids_gt: np.ndarray) -> Dict[int, int]:
    return {int(cid): i for i, cid in enumerate(case_ids_gt)}


# ---------------------------------------------------------------------------
# Arc-length error computation
# ---------------------------------------------------------------------------

def compute_arclength_position_errors(
    shapes: Dict[str, dict],
    positions_gt: np.ndarray,
    case_ids_gt: np.ndarray,
) -> Dict[str, np.ndarray]:
    cmap = _cid_map(case_ids_gt)
    out: Dict[str, np.ndarray] = {}
    for lname, d in shapes.items():
        n, M, _ = d["p_est"].shape
        errs = np.full((n, M), np.nan)
        for k in range(n):
            gt_idx = cmap.get(int(d["case_ids"][k]))
            if gt_idx is None:
                continue
            errs[k] = np.linalg.norm(d["p_est"][k] - positions_gt[gt_idx], axis=1)
        out[lname] = errs
    return out


def compute_arclength_orientation_errors(
    shapes: Dict[str, dict],
    orientations_gt: np.ndarray,
    case_ids_gt: np.ndarray,
) -> Dict[str, np.ndarray]:
    cmap = _cid_map(case_ids_gt)
    out: Dict[str, np.ndarray] = {}
    for lname, d in shapes.items():
        if "R_est" not in d:
            raise RuntimeError(
                f"R_est missing for '{lname}'. "
                "Re-run evaluate_kirchhoff_shape_estimation.py."
            )
        n, M, _, _ = d["R_est"].shape
        errs = np.full((n, M), np.nan)
        for k in range(n):
            gt_idx = cmap.get(int(d["case_ids"][k]))
            if gt_idx is None:
                continue
            for i in range(M):
                errs[k, i] = np.degrees(
                    np.linalg.norm(_so3_log(d["R_est"][k, i] @ orientations_gt[gt_idx, i].T))
                )
        out[lname] = errs
    return out


def _case_arclength_errors(
    cid: int,
    shapes: Dict[str, dict],
    positions_gt: np.ndarray,
    orientations_gt: np.ndarray,
    cmap_gt: Dict[int, int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    gt_idx = cmap_gt[cid]
    p_gt   = positions_gt[gt_idx]
    R_gt   = orientations_gt[gt_idx]
    pos_err: Dict[str, np.ndarray] = {}
    ori_err: Dict[str, np.ndarray] = {}
    for lname, d in shapes.items():
        mask  = d["case_ids"] == cid
        p_est = d["p_est"][mask]
        pos_err[lname] = np.linalg.norm(p_est - p_gt[None], axis=2)
        if "R_est" in d:
            R_est = d["R_est"][mask]
            n_n, M_ = R_est.shape[:2]
            o = np.zeros((n_n, M_))
            for r in range(n_n):
                for i in range(M_):
                    o[r, i] = np.degrees(
                        np.linalg.norm(_so3_log(R_est[r, i] @ R_gt[i].T))
                    )
            ori_err[lname] = o
    return pos_err, ori_err


# ---------------------------------------------------------------------------
# Complexity / nonplanarity  (for representative-case selection)
# ---------------------------------------------------------------------------

def _compute_complexity(positions_gt: np.ndarray) -> np.ndarray:
    def _n01(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)

    tip_lat = np.linalg.norm(positions_gt[:, -1, :2], axis=1)
    max_lat = np.linalg.norm(positions_gt[:, :, :2], axis=2).max(1)
    dp  = np.diff(positions_gt, axis=1)
    dp_u = dp / np.where(np.linalg.norm(dp, axis=2, keepdims=True) < 1e-12,
                         1.0, np.linalg.norm(dp, axis=2, keepdims=True))
    cos_a = np.clip((dp_u[:, :-1] * dp_u[:, 1:]).sum(2), -1.0, 1.0)
    curv  = np.degrees(np.arccos(cos_a)).sum(1)
    pts_c = positions_gt - positions_gt.mean(axis=1, keepdims=True)
    _, sv, _ = np.linalg.svd(pts_c, full_matrices=False)
    nonplan = sv[:, 2] / (sv.sum(axis=1) + 1e-12)
    return (0.30 * _n01(tip_lat) + 0.25 * _n01(max_lat)
            + 0.25 * _n01(curv)  + 0.20 * _n01(nonplan))


def _nonplanarity_score(positions_gt: np.ndarray) -> np.ndarray:
    pts_c = positions_gt - positions_gt.mean(axis=1, keepdims=True)
    _, sv, _ = np.linalg.svd(pts_c, full_matrices=False)
    return sv[:, 2] / (sv.sum(axis=1) + 1e-12)


def _per_case_mean_errors(per_case: List[dict]) -> Dict[Tuple[int, str], float]:
    acc: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for row in per_case:
        acc[(row["case_id"], row["layout_name"])].append(row["mean_centerline_error"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def compute_representative_cases(
    positions_gt: np.ndarray,
    case_ids_gt: np.ndarray,
    layout_names: List[str],
    per_case: List[dict],
    n_cases: int = 4,
    np_percentile: float = 40.0,
) -> List[Tuple[int, str, float, float]]:
    """
    Select n_cases nonplanar-and-complex representative shapes.

    1. Filter to cases with nonplanarity >= np_percentile-th percentile
       (keeps the top (100 - np_percentile) % by nonplanarity).
    2. Within that set divide by complexity into n_cases equal bands.
    3. From each band pick the case with the largest inter-layout error gap.

    Returns [(case_id, complexity_label, cx_score, np_score), ...].
    """
    complexity = _compute_complexity(positions_gt)
    nonplan    = _nonplanarity_score(positions_gt)
    err_map    = _per_case_mean_errors(per_case)

    np_thresh   = np.percentile(nonplan, np_percentile)
    filt_idxs   = np.where(nonplan >= np_thresh)[0]
    sorted_filt = filt_idxs[np.argsort(complexity[filt_idxs])]
    N_filt      = len(sorted_filt)
    labels      = _CX_LABELS.get(n_cases, [f"cx {i+1}/{n_cases}" for i in range(n_cases)])

    out = []
    for gi in range(n_cases):
        start = gi * N_filt // n_cases
        end   = (gi + 1) * N_filt // n_cases
        cands = sorted_filt[start:end]
        if len(cands) == 0:
            cands = sorted_filt

        if len(layout_names) >= 2:
            divs = [abs(err_map.get((int(case_ids_gt[i]), layout_names[0]),  0.0)
                      - err_map.get((int(case_ids_gt[i]), layout_names[-1]), 0.0))
                    for i in cands]
            best = cands[int(np.argmax(divs))]
        else:
            best = cands[len(cands) // 2]

        cid = int(case_ids_gt[best])
        out.append((cid, labels[gi],
                    float(complexity[best]), float(nonplan[best])))
    return out


# ---------------------------------------------------------------------------
# 3-D rendering helpers  (adapted from archive script)
# ---------------------------------------------------------------------------

def _disk_ring(
    ax,
    origin: np.ndarray,
    R: np.ndarray,
    radius: float = _DISK_RADIUS,
    height: float = _DISK_HEIGHT,
    color: str = _DISK_COLOR,
    alpha: float = _DISK_ALPHA,
    n_pts: int = 36,
) -> None:
    """
    Draw a thin disk (filled cylinder) centred at `origin`, oriented by `R`.
    The disk plane is perpendicular to R[:,2] (the tangent / z-axis).
    Adapted from plot_cylinder() in 3d_CR_with_disc_tendon_v2.py.
    """
    theta = np.linspace(0, 2 * np.pi, n_pts)
    z_vals = np.array([-height / 2, height / 2])
    tg, zg = np.meshgrid(theta, z_vals)
    xg = radius * np.cos(tg)
    yg = radius * np.sin(tg)
    pts = np.vstack((xg.ravel(), yg.ravel(), zg.ravel()))
    pts_w = (R @ pts) + origin[:, None]
    X = pts_w[0].reshape(2, n_pts)
    Y = pts_w[1].reshape(2, n_pts)
    Z = pts_w[2].reshape(2, n_pts)
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha,
                    linewidth=0, shade=False, antialiased=False)

    # draw top and bottom rim circles
    for s_z in (-height / 2, height / 2):
        rim = R @ np.vstack((radius * np.cos(theta),
                             radius * np.sin(theta),
                             np.full_like(theta, s_z))) + origin[:, None]
        ax.plot(rim[0], rim[1], rim[2], color="dimgrey",
                linewidth=0.6, alpha=_DISK_EDGE_ALPHA, zorder=1)


def _draw_frame(
    ax,
    origin: np.ndarray,
    R: np.ndarray,
    scale: float = _FRAME_SCALE,
    lw: float = _FRAME_LW,
    alpha: float = _FRAME_ALPHA_GT,
) -> None:
    """Draw an RGB quiver frame at `origin` oriented by `R`."""
    for j, col in enumerate(_FRAME_COLS):
        d = R[:, j] * scale
        ax.quiver(origin[0], origin[1], origin[2],
                  d[0], d[1], d[2],
                  color=col, linewidth=lw, alpha=alpha,
                  arrow_length_ratio=0.30, normalize=False)


def _draw_base_plate(
    ax,
    origin: np.ndarray,
    R: np.ndarray,
    half_side: float = 0.010,
    color: str = "#3a3a3a",
    alpha: float = 0.72,
) -> None:
    """
    Draw a small square base plate at the fixed end (s = 0).
    The plate lies in the base-frame x-y plane (R[:,0] × R[:,1]).
    Rendered as a filled Poly3DCollection quad.
    """
    x_ax = R[:, 0] * half_side
    y_ax = R[:, 1] * half_side
    corners = np.array([
        origin - x_ax - y_ax,
        origin + x_ax - y_ax,
        origin + x_ax + y_ax,
        origin - x_ax + y_ax,
    ])
    poly = Poly3DCollection([corners], alpha=alpha, zorder=2)
    poly.set_facecolor(color)
    poly.set_edgecolor("#111111")
    poly.set_linewidth(0.9)
    ax.add_collection3d(poly)


def render_robot_like_gt(
    ax,
    positions: np.ndarray,
    orientations: np.ndarray,
    s_grid: np.ndarray,
    frame_s_locs: Tuple[float, ...] = _FRAME_S_LOCS,
) -> None:
    """
    Render the GT backbone with sparse RGB frames.
    `positions`   : (M, 3)   backbone positions
    `orientations`: (M, 3, 3) rotation matrices
    `s_grid`      : (M,)     normalised arc-length values
    """
    # base plate at the fixed end (s = 0)
    _draw_base_plate(ax, positions[0], orientations[0])

    # backbone
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            "k-", lw=2.2, zorder=5, label="GT (Kirchhoff)")

    # sparse frames along GT backbone
    for s_val in frame_s_locs:
        idx = int(np.argmin(np.abs(s_grid - s_val)))
        _draw_frame(ax, positions[idx], orientations[idx],
                    alpha=_FRAME_ALPHA_GT)


def draw_sparse_frames(
    ax,
    positions: np.ndarray,
    orientations: np.ndarray,
    s_grid: np.ndarray,
    frame_s_locs: Tuple[float, ...] = _FRAME_S_LOCS,
    alpha: float = _FRAME_ALPHA_EST,
    scale: float = _FRAME_SCALE * 0.85,
) -> None:
    """Draw lighter frames for an estimated trajectory (e.g. 3-IMU)."""
    for s_val in frame_s_locs:
        idx = int(np.argmin(np.abs(s_grid - s_val)))
        _draw_frame(ax, positions[idx], orientations[idx],
                    scale=scale, alpha=alpha)


def _clean_3d_axis(ax) -> None:
    """
    Remove panes, ticks, labels, and gridlines.
    Adapted from the axis-off block in 3d_CR_with_disc_tendon_v2.py.
    """
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("");  ax.set_ylabel("");  ax.set_zlabel("")
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["axisline"]["linewidth"] = 0
        axis._axinfo["grid"]["linewidth"] = 0
        axis.set_ticklabels([])


def _tight_axis_limits(ax, all_pts: np.ndarray, pad_frac: float = 0.03) -> None:
    """
    Set tight 3-D limits and shape the view-box to match the data span.
    Adapted from set_equal_axis_scale() in 3d_CR_with_disc_tendon_v2.py.

    Using the actual per-axis span ratios for set_box_aspect prevents the
    equal-cube box from leaving large whitespace when the robot is elongated
    along one axis.  Spans are clamped to >= 20 % of the max span so no axis
    collapses to a sliver.
    """
    lo, hi   = all_pts.min(0), all_pts.max(0)
    span     = hi - lo
    max_span = span.max()
    pad      = max(max_span * pad_frac, 1e-3)
    ctr      = (lo + hi) / 2
    half     = max_span / 2 + pad
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    # box aspect proportional to data span → shape fills the subplot
    ratios = np.clip(span / (max_span + 1e-12), 0.20, 1.0)
    try:
        ax.set_box_aspect(ratios.tolist())
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Panel helpers  (2-D panels shared by both figures)
# ---------------------------------------------------------------------------

def _add_imu_markers(ax, layout_names: List[str]) -> None:
    done: set = set()
    for lname in layout_names:
        for sp in _IMU_S.get(lname, []):
            if sp not in done:
                ax.axvline(sp, color="grey", lw=0.7, ls=":", alpha=0.55, zorder=0)
                done.add(sp)


def _panel_violin(
    ax,
    layout_names: List[str],
    metric_keys: List[str],
    per_case: List[dict],
    scale_mm: bool,
    ylabel: str,
    metric_xlabels: List[str],
) -> None:
    for mi, mkey in enumerate(metric_keys):
        cx = _GROUP_CENTRES[mi]
        for li, lname in enumerate(layout_names):
            vals = np.array([r[mkey] for r in per_case
                             if r["layout_name"] == lname])
            if scale_mm:
                vals = vals * 1e3
            xpos = cx + (li - 0.5) * 2 * _HALF_GAP
            parts = ax.violinplot([vals], positions=[xpos], widths=_VWIDTH,
                                  showmeans=False, showmedians=False,
                                  showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(COLORS[lname])
                body.set_edgecolor("none")
                body.set_alpha(0.60)
            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            ax.vlines(xpos, q25, q75, color=COLORS[lname], linewidth=3.5, zorder=5)
            ax.scatter([xpos], [med], color="white", s=32, zorder=6,
                       edgecolors="grey", linewidths=0.7)

    ax.set_xticks(_GROUP_CENTRES[: len(metric_keys)])
    ax.set_xticklabels(metric_xlabels, fontsize=TICK_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.yaxis.grid(True, alpha=0.30, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlim(_GROUP_CENTRES[0] - 0.70,
                _GROUP_CENTRES[len(metric_keys) - 1] + 0.70)
    ax.set_ylim(bottom=0)


def _panel_arclength_global(
    ax,
    errors: Dict[str, np.ndarray],
    s_grid: np.ndarray,
    layout_names: List[str],
    ylabel: str,
    scale_mm: bool = False,
) -> None:
    for lname in layout_names:
        e    = errors[lname] * (1e3 if scale_mm else 1.0)
        mean = np.nanmean(e,           axis=0)
        p25  = np.nanpercentile(e, 25, axis=0)
        p75  = np.nanpercentile(e, 75, axis=0)
        col  = COLORS[lname]
        ax.plot(s_grid, mean, color=col, lw=LW, label=lname)
        ax.fill_between(s_grid, p25, p75, color=col, alpha=0.18, linewidth=0)
    _add_imu_markers(ax, layout_names)
    ax.set_xlabel("Normalised arc-length  $s$", fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_xlim(s_grid[0], s_grid[-1])
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, alpha=0.30, linestyle="--")
    ax.set_axisbelow(True)


def _panel_arclength_case(
    ax,
    errors: Dict[str, np.ndarray],
    s_grid: np.ndarray,
    layout_names: List[str],
    ylabel: str,
    scale_mm: bool = False,
    show_xlabel: bool = True,
) -> None:
    for lname in layout_names:
        e    = errors[lname] * (1e3 if scale_mm else 1.0)
        mean = np.nanmean(e,           axis=0)
        p25  = np.nanpercentile(e, 25, axis=0)
        p75  = np.nanpercentile(e, 75, axis=0)
        col  = COLORS[lname]
        ax.plot(s_grid, mean, color=col, lw=LW, ls=LS[lname])
        ax.fill_between(s_grid, p25, p75, color=col, alpha=0.20, linewidth=0)
    _add_imu_markers(ax, layout_names)
    if show_xlabel:
        ax.set_xlabel("Arc-length  $s$", fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS - 1)
    ax.set_xlim(s_grid[0], s_grid[-1])
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, alpha=0.30, linestyle="--")
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Figure 1 — Main aggregate  (2 × 2)
# ---------------------------------------------------------------------------

def build_main_aggregate_figure(
    per_case: List[dict],
    layout_names: List[str],
    pos_errors: Dict[str, np.ndarray],
    ori_errors: Dict[str, np.ndarray],
    s_grid: np.ndarray,
    n_cases: int,
    n_noise: int,
    save_stem: Optional[Path] = None,
) -> None:
    """2×2 publication-quality aggregate figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    _panel_violin(ax_a, layout_names,
                  ["mean_centerline_error", "tip_position_error"],
                  per_case, scale_mm=True, ylabel="Error  [mm]",
                  metric_xlabels=["Mean centerline\nerror", "Tip position\nerror"])

    _panel_violin(ax_b, layout_names,
                  ["mean_orientation_error_deg", "tip_orientation_error_deg"],
                  per_case, scale_mm=False, ylabel="Error  [deg]",
                  metric_xlabels=["Mean orientation\nerror", "Tip orientation\nerror"])

    _panel_arclength_global(ax_c, pos_errors, s_grid, layout_names,
                            ylabel="Position error  [mm]", scale_mm=True)

    _panel_arclength_global(ax_d, ori_errors, s_grid, layout_names,
                            ylabel="Orientation error  [deg]")

    panel_titles = [
        "Position metrics",
        "Orientation metrics",
        "Position error along arc-length",
        "Orientation error along arc-length",
    ]
    for ax, lbl, ttl in zip(
        [ax_a, ax_b, ax_c, ax_d],
        ["(a)", "(b)", "(c)", "(d)"],
        panel_titles,
    ):
        ax.text(0.02, 0.98, lbl, transform=ax.transAxes,
                fontsize=PANEL_FS, fontweight="bold", va="top", ha="left")
        ax.set_title(ttl, fontsize=TITLE_FS, pad=4)

    proxies = [mpatches.Patch(facecolor=COLORS[nm], alpha=0.75, label=nm)
               for nm in layout_names]
    fig.legend(handles=proxies, loc="lower center", ncol=len(layout_names),
               fontsize=LEGEND_FS, framealpha=0.9, bbox_to_anchor=(0.5, 0.00))

    ax_d.text(0.98, 0.97,
              f"n = {n_cases} cases  x  {n_noise} realisations",
              transform=ax_d.transAxes, fontsize=9,
              ha="right", va="top", color="grey", style="italic")

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_stem is not None:
        for ext in ("png", "pdf"):
            p = save_stem.parent / (save_stem.name + f".{ext}")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved -> {p}")

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Representative-case detail  (n_cases cols × 3 rows)
# ---------------------------------------------------------------------------

def build_representative_case_figure(
    selected: List[Tuple[int, str, float, float]],
    positions_gt: np.ndarray,
    orientations_gt: np.ndarray,
    case_ids_gt: np.ndarray,
    layout_names: List[str],
    shapes: Dict[str, dict],
    s_grid: np.ndarray,
    elev: float = _ELEV,
    azim: float = _AZIM,
    save_stem: Optional[Path] = None,
) -> None:
    """
    n_cases × 3 representative-case figure with robot-style 3D rendering.

    Row 0: 3-D overlays  (GT backbone + disks + frames, EKF overlays)
    Row 1: per-case arc-length position error  [mm]
    Row 2: per-case arc-length orientation error  [deg]
    """
    n_cols  = len(selected)
    cmap_gt = _cid_map(case_ids_gt)

    # ── figure layout ───────────────────────────────────────────────────────
    fig_w = max(5.5 * n_cols, 18.0)
    fig = plt.figure(figsize=(fig_w, 13.5))
    gs  = fig.add_gridspec(
        3, n_cols,
        height_ratios=[5.5, 2.0, 2.0],
        hspace=0.25, wspace=0.18,
        left=0.06, right=0.97, top=0.95, bottom=0.07,
    )

    legend_handles: list = []
    legend_labels:  list = []
    legend_done = False

    for col, entry in enumerate(selected):
        cid, cx_lbl, cx_sc, np_sc = entry[0], entry[1], entry[2], entry[3]
        gt_idx = cmap_gt[cid]
        p_gt   = positions_gt[gt_idx]    # (M, 3)
        R_gt   = orientations_gt[gt_idx] # (M, 3, 3)

        # ── Row 0: robot-style 3D overlay ───────────────────────────────────
        ax3 = fig.add_subplot(gs[0, col], projection="3d")

        # GT: backbone + disks + frames
        render_robot_like_gt(ax3, p_gt, R_gt, s_grid)
        if not legend_done:
            legend_handles.append(
                plt.Line2D([0], [0], color="k", lw=2.0, label="GT (Kirchhoff)")
            )
            legend_labels.append("GT (Kirchhoff)")

        # EKF estimates
        for lname in layout_names:
            d    = shapes[lname]
            mask = (d["case_ids"] == cid) & (d["noise_real"] == 0)
            if not np.any(mask):
                continue
            p_est = d["p_est"][mask][0]
            ax3.plot(p_est[:, 0], p_est[:, 1], p_est[:, 2],
                     color=COLORS[lname], ls=LS[lname], lw=LW * 0.95,
                     alpha=0.88, label=lname, zorder=6)
            if not legend_done:
                legend_handles.append(
                    plt.Line2D([0], [0], color=COLORS[lname], ls=LS[lname],
                               lw=LW, label=lname)
                )
                legend_labels.append(lname)

            # optional 3-IMU frames (lighter)
            if lname == "3-IMU" and "R_est" in d:
                R_est = d["R_est"][mask][0]
                draw_sparse_frames(ax3, p_est, R_est, s_grid,
                                   alpha=_FRAME_ALPHA_EST)

        legend_done = True

        # per-panel tight axis limits (each case fills its own panel)
        panel_pts = [p_gt]
        for d in shapes.values():
            mask_p = (d["case_ids"] == cid) & (d["noise_real"] == 0)
            if np.any(mask_p):
                panel_pts.append(d["p_est"][mask_p][0])
        _tight_axis_limits(ax3, np.vstack(panel_pts))
        ax3.view_init(elev=elev, azim=azim)
        try:
            ax3.set_proj_type("ortho")
        except Exception:
            pass
        _clean_3d_axis(ax3)

        # panel title — clean, no shorthand
        letter = chr(ord("a") + col)
        ax3.set_title(f"({letter})  {cx_lbl}", fontsize=TITLE_FS, pad=6)

        # minimal annotation: case ID only
        ax3.text2D(0.04, 0.96, f"Case {cid}",
                   transform=ax3.transAxes, fontsize=ANN_FS,
                   va="top", ha="left",
                   bbox=dict(facecolor="white", alpha=0.60,
                             edgecolor="none", pad=2))

        # ── Row 1: per-case position error ──────────────────────────────────
        ax_p = fig.add_subplot(gs[1, col])
        pe, oe = _case_arclength_errors(
            cid, shapes, positions_gt, orientations_gt, cmap_gt
        )
        _panel_arclength_case(ax_p, pe, s_grid, layout_names,
                              ylabel="Pos. err.  [mm]",
                              scale_mm=True, show_xlabel=False)
        if col > 0:
            ax_p.set_ylabel("")

        # ── Row 2: per-case orientation error ───────────────────────────────
        ax_o = fig.add_subplot(gs[2, col])
        _panel_arclength_case(ax_o, oe, s_grid, layout_names,
                              ylabel="Ori. err.  [deg]",
                              scale_mm=False, show_xlabel=True)
        if col > 0:
            ax_o.set_ylabel("")

    # ── shared legend below figure ──────────────────────────────────────────
    gt_proxy = plt.Line2D([0], [0], color="k", lw=2.0, label="GT (Kirchhoff)")
    est_proxies = [
        mpatches.Patch(facecolor=COLORS[nm], alpha=0.75, label=nm)
        for nm in layout_names
    ]
    fig.legend(
        handles=[gt_proxy] + est_proxies,
        loc="lower center",
        ncol=1 + len(layout_names),
        fontsize=LEGEND_FS,
        framealpha=0.92,
        bbox_to_anchor=(0.5, 0.01),
    )

    if save_stem is not None:
        for ext in ("png", "pdf"):
            p = save_stem.parent / (save_stem.name + f".{ext}")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved -> {p}")

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate both main-text cross-model figures (no EKF re-run).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default="gt_data/results",
                        help="directory containing saved JSON and NPZ results")
    parser.add_argument("--gt", default="gt_data/kirchhoff_gt_dataset.npz",
                        help="path to GT dataset NPZ")
    parser.add_argument("--plot-aggregate-only", action="store_true", default=False,
                        help="generate Figure 1 (aggregate) only")
    parser.add_argument("--plot-representative-only", action="store_true", default=False,
                        help="generate Figure 2 (representative cases) only")
    parser.add_argument("--n-rep-cases", type=int, default=4,
                        help="number of representative cases in Figure 2")
    parser.add_argument("--np-percentile", type=float, default=40.0,
                        help="nonplanarity filter: keep cases >= this percentile "
                             "(0 = all, 40 = top 60%% most nonplanar)")
    parser.add_argument("--elev", type=float, default=_ELEV,
                        help="3-D view elevation angle")
    parser.add_argument("--azim", type=float, default=_AZIM,
                        help="3-D view azimuth angle")
    parser.add_argument("--no-save", action="store_true",
                        help="skip saving output files")
    args = parser.parse_args()

    # default: generate both figures
    if not args.plot_aggregate_only and not args.plot_representative_only:
        args.plot_aggregate_only      = True
        args.plot_representative_only = True

    results_dir = Path(args.results_dir)
    gt_path     = Path(args.gt)

    # ── load all saved data ─────────────────────────────────────────────────
    bundle = load_saved_results(results_dir, gt_path)
    per_case        = bundle["per_case"]
    layout_names    = bundle["layout_names"]
    positions_gt    = bundle["positions_gt"]
    orientations_gt = bundle["orientations_gt"]
    case_ids_gt     = bundle["case_ids_gt"]
    shapes          = bundle["shapes"]
    n_cases         = bundle["n_cases"]
    n_noise         = bundle["n_noise"]
    s_grid          = bundle["s_grid"]

    # ── arc-length errors ───────────────────────────────────────────────────
    print("Computing arc-length position errors ...")
    pos_errors = compute_arclength_position_errors(shapes, positions_gt, case_ids_gt)
    print("Computing arc-length SO(3) orientation errors ...")
    ori_errors = compute_arclength_orientation_errors(
        shapes, orientations_gt, case_ids_gt
    )

    print(f"\nData scope: {n_cases} cases x {n_noise} realisations per layout")
    for lname in layout_names:
        me = np.nanmean(pos_errors[lname]) * 1e3
        oe = np.nanmean(ori_errors[lname])
        print(f"  {lname:<8s}  pos={me:.3f} mm   ori={oe:.3f} deg")

    # ── Figure 1: main aggregate ────────────────────────────────────────────
    if args.plot_aggregate_only:
        stem1 = (None if args.no_save
                 else results_dir / "kirchhoff_shape_est_main_aggregate_violin_v2")
        print("\n--- Figure 1: 2x2 aggregate ---")
        build_main_aggregate_figure(
            per_case=per_case,
            layout_names=layout_names,
            pos_errors=pos_errors,
            ori_errors=ori_errors,
            s_grid=s_grid,
            n_cases=n_cases,
            n_noise=n_noise,
            save_stem=stem1,
        )

    # ── Figure 2: representative cases ─────────────────────────────────────
    if args.plot_representative_only:
        stem2 = (None if args.no_save
                 else results_dir / "kirchhoff_shape_est_composite_maintext_v3")
        print(f"\n--- Figure 2: representative-case detail "
              f"({args.n_rep_cases} cases, "
              f"top-{100 - args.np_percentile:.0f}% nonplanar) ---")
        selected = compute_representative_cases(
            positions_gt, case_ids_gt, layout_names, per_case,
            n_cases=args.n_rep_cases,
            np_percentile=args.np_percentile,
        )
        err_map = _per_case_mean_errors(per_case)
        print("  Selected cases:")
        for cid, lbl, cx, np_s in selected:
            e_str = "  ".join(
                f"{ln}={err_map.get((cid, ln), float('nan')) * 1e3:.2f}mm"
                for ln in layout_names
            )
            print(f"    [{lbl}]  case {cid:3d}  cx={cx:.3f}  np={np_s:.4f}  {e_str}")

        build_representative_case_figure(
            selected=selected,
            positions_gt=positions_gt,
            orientations_gt=orientations_gt,
            case_ids_gt=case_ids_gt,
            layout_names=layout_names,
            shapes=shapes,
            s_grid=s_grid,
            elev=args.elev,
            azim=args.azim,
            save_stem=stem2,
        )


if __name__ == "__main__":
    main()
