#!/usr/bin/env python3
"""
Plot-only companion to evaluate_kirchhoff_shape_estimation.py.

Reads saved results (JSON + shapes NPZ + GT NPZ) and regenerates all figures
without re-running the EKF.

Figure types
------------
  --plot-summary           bar chart: mean centerline error & tip error
                           → kirchhoff_shape_est_summary.{png,pdf}

  --plot-metrics           per-layout violin plots (all 5 geometry metrics)
                           → kirchhoff_shape_est_metrics.{png,pdf}

  --plot-arclength-pos     position error vs normalised arc-length s
                           → kirchhoff_shape_est_arclength_pos.{png,pdf}

  --plot-arclength-ori     tangent-direction error vs arc-length s
                           (approximates SO(3) z-axis error; full R_est not
                            saved — see note in plot_arclength_orientation_errors)
                           → kirchhoff_shape_est_arclength_ori.{png,pdf}

  --plot-complexity-overlays
                           2×3 shape overlays selected by shape complexity
                           → kirchhoff_shape_est_overlays_complexity.{png,pdf}

  --plot-overlays          original low/median/high error overlays (legacy)
                           → kirchhoff_shape_est_overlays.{png,pdf}

Usage
-----
  python plot_kirchhoff_shape_est.py
  python plot_kirchhoff_shape_est.py --results-dir gt_data/results --plot-summary
  python plot_kirchhoff_shape_est.py --plot-arclength-pos --plot-arclength-ori
  python plot_kirchhoff_shape_est.py --plot-complexity-overlays
  python plot_kirchhoff_shape_est.py --plot-all
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# SO(3) helper  (identical to evaluate_kirchhoff_shape_estimation.py)
# ---------------------------------------------------------------------------

def _vee(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def so3_log(R: np.ndarray) -> np.ndarray:
    val   = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(val)
    if abs(theta) < 1e-8:
        return _vee(R - R.T) / 2.0
    return (theta / (2.0 * np.sin(theta))) * _vee(R - R.T)


# ---------------------------------------------------------------------------
# Style constants  (publication-friendly)
# ---------------------------------------------------------------------------

LABEL_FS  = 12
TICK_FS   = 11
LEGEND_FS = 11
TITLE_FS  = 12
ANN_FS    = 7

# Orientation frame drawing
FRAME_S_LOCS      = (0.25, 0.50, 0.75, 1.00)
FRAME_SCALE       = 0.003            # quiver arrow length [m] — ~3 % of 0.10 m rod
FRAME_LW          = 0.9
_FRAME_AXIS_COLS  = ("r", "g", "b")  # x / y / z axis colours
_ERROR_CMAP       = "plasma"         # colourmap for position-error curve

# Per-layout plot style — extend for more than 2 layouts
_PALETTE = [
    {"color": "#1f77b4", "ls": "--",  "lw": 1.8},   # blue
    {"color": "#d62728", "ls": "-.",  "lw": 1.8},   # red
    {"color": "#2ca02c", "ls": ":",   "lw": 1.8},   # green
    {"color": "#9467bd", "ls": "--",  "lw": 1.8},   # purple
]


def _lstyle(layout_name: str, layout_names: List[str]) -> dict:
    idx = layout_names.index(layout_name) if layout_name in layout_names else 0
    return _PALETTE[idx % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_gt(gt_npz: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns positions (N,M,3), orientations (N,M,3,3), case_ids (N,)."""
    data = np.load(gt_npz, allow_pickle=True)
    N, M, _ = data["positions"].shape
    ori = data["orientations"].reshape(N, M, 3, 3)
    return data["positions"], ori, data["case_id"]


def load_shapes(shapes_npz: Path) -> Tuple[List[str], Dict[str, dict]]:
    """
    Returns (layout_names, shapes_dict).
    shapes_dict[lname] keys: p_est (n,M,3), R_est (n,M,3,3) if saved,
    case_ids (n,), noise_real (n,).
    """
    data  = np.load(shapes_npz, allow_pickle=True)
    names = list(data["layout_names"])
    out: Dict[str, dict] = {}
    for nm in names:
        entry: dict = {
            "p_est"     : data[f"p_est_{nm}"],
            "case_ids"  : data[f"case_ids_{nm}"],
            "noise_real": data[f"noise_real_{nm}"],
        }
        if f"R_est_{nm}" in data.files:
            entry["R_est"] = data[f"R_est_{nm}"]
        out[nm] = entry
    return names, out


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _cid_map(case_ids_gt: np.ndarray) -> Dict[int, int]:
    return {int(cid): i for i, cid in enumerate(case_ids_gt)}



def _tight_axis_limits(ax, pts_list: list, pad_frac: float = 0.07) -> None:
    all_pts = np.vstack(pts_list)
    lo, hi  = all_pts.min(0), all_pts.max(0)
    span    = hi - lo
    pad     = np.where(span > 0, span * pad_frac, 1e-3)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])


def _save_show(fig, save_stem: Optional[Path], suffix: str) -> None:
    plt.tight_layout()
    plt.show()
    if save_stem is not None:
        for ext in ("png", "pdf"):
            p = save_stem.parent / (save_stem.name + f"_{suffix}.{ext}")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Figure saved → {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# A  Arc-length position error
# ---------------------------------------------------------------------------

def compute_arclength_position_errors(
    shapes: Dict[str, dict],
    positions_gt: np.ndarray,
    case_ids_gt: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Returns {layout_name: errors (n_rows, M)} in metres.
    Rows with no matching GT case are filled with NaN.
    """
    cmap = _cid_map(case_ids_gt)
    out: Dict[str, np.ndarray] = {}
    for lname, d in shapes.items():
        n, M, _ = d["p_est"].shape
        errs = np.full((n, M), np.nan)
        for k in range(n):
            gt_idx = cmap.get(int(d["case_ids"][k]))
            if gt_idx is None:
                continue
            diff       = d["p_est"][k] - positions_gt[gt_idx]   # (M, 3)
            errs[k]    = np.linalg.norm(diff, axis=1)
        out[lname] = errs
    return out


def plot_arclength_position_errors(
    pos_errors: Dict[str, np.ndarray],
    s_grid: np.ndarray,
    layout_names: List[str],
    save_stem: Optional[Path] = None,
) -> None:
    """
    Position error || p_est(s) - p_gt(s) || vs normalised arc-length.
    Mean curve + 25–75 percentile shaded band.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for lname in layout_names:
        errs  = pos_errors[lname] * 1e3          # m → mm
        mean  = np.nanmean(errs,                  axis=0)
        p25   = np.nanpercentile(errs,  25,       axis=0)
        p75   = np.nanpercentile(errs,  75,       axis=0)
        st    = _lstyle(lname, layout_names)
        ax.plot(s_grid, mean,
                color=st["color"], ls=st["ls"], lw=st["lw"], label=lname)
        ax.fill_between(s_grid, p25, p75,
                        color=st["color"], alpha=0.18, linewidth=0)

    ax.set_xlabel("Normalised arc-length  $s$", fontsize=LABEL_FS)
    ax.set_ylabel("Position error  [mm]",       fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_xlim(s_grid[0], s_grid[-1])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=LEGEND_FS)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    # sensor markers: IMU positions inferred from layout names
    _mark_imu_positions(ax, layout_names, pos_errors)

    _save_show(fig, save_stem, "arclength_pos")


def _mark_imu_positions(ax, layout_names, errs_dict) -> None:
    """Add light vertical lines at common IMU positions if consistent across layouts."""
    # Heuristic: snap to nearest 0.25 fraction of s based on layout name
    imu_s_hints = {"2-IMU": [0.50, 1.00], "3-IMU": [0.25, 0.50, 1.00]}
    done: set = set()
    for lname in layout_names:
        for s_pos in imu_s_hints.get(lname, []):
            if s_pos not in done:
                ax.axvline(s_pos, color="grey", lw=0.7, ls=":", alpha=0.6, zorder=0)
                done.add(s_pos)


# ---------------------------------------------------------------------------
# B  Arc-length SO(3) orientation error
# ---------------------------------------------------------------------------

def compute_arclength_orientation_errors(
    shapes: Dict[str, dict],
    orientations_gt: np.ndarray,
    case_ids_gt: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Full SO(3) orientation error at each arc-length point.

        e_R(s) = || log( R_est(s) @ R_gt(s)^T ) ||   [rad → deg]

    Requires R_est to be present in the shapes dict (saved by
    evaluate_kirchhoff_shape_estimation.py).  Raises RuntimeError if missing.

    Returns {layout_name: errors_deg (n_rows, M)}.
    """
    cmap = _cid_map(case_ids_gt)
    out: Dict[str, np.ndarray] = {}
    for lname, d in shapes.items():
        if "R_est" not in d:
            raise RuntimeError(
                f"R_est not found for layout '{lname}' in the shapes NPZ.\n"
                "Re-run evaluate_kirchhoff_shape_estimation.py to regenerate "
                "kirchhoff_shape_est_shapes.npz with R_est arrays."
            )
        n, M, _, _ = d["R_est"].shape
        errs = np.full((n, M), np.nan)
        for k in range(n):
            gt_idx = cmap.get(int(d["case_ids"][k]))
            if gt_idx is None:
                continue
            R_est_k = d["R_est"][k]              # (M, 3, 3)
            R_gt_k  = orientations_gt[gt_idx]    # (M, 3, 3)
            for i in range(M):
                errs[k, i] = np.degrees(
                    np.linalg.norm(so3_log(R_est_k[i] @ R_gt_k[i].T))
                )
        out[lname] = errs
    return out


def plot_arclength_orientation_errors(
    ori_errors: Dict[str, np.ndarray],
    s_grid: np.ndarray,
    layout_names: List[str],
    save_stem: Optional[Path] = None,
) -> None:
    """
    SO(3) orientation error [deg] vs normalised arc-length.
    Mean curve + 25–75 percentile shaded band.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for lname in layout_names:
        errs = ori_errors[lname]
        mean = np.nanmean(errs,           axis=0)
        p25  = np.nanpercentile(errs, 25, axis=0)
        p75  = np.nanpercentile(errs, 75, axis=0)
        st   = _lstyle(lname, layout_names)
        ax.plot(s_grid, mean,
                color=st["color"], ls=st["ls"], lw=st["lw"], label=lname)
        ax.fill_between(s_grid, p25, p75,
                        color=st["color"], alpha=0.18, linewidth=0)

    ax.set_xlabel("Normalised arc-length  $s$",       fontsize=LABEL_FS)
    ax.set_ylabel("Orientation error  [deg]",         fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_xlim(s_grid[0], s_grid[-1])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=LEGEND_FS)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    _mark_imu_positions(ax, layout_names, ori_errors)

    _save_show(fig, save_stem, "arclength_ori")


# ---------------------------------------------------------------------------
# C  Shape complexity scoring & representative-case selection
# ---------------------------------------------------------------------------

def compute_shape_complexity_scores(positions_gt: np.ndarray) -> np.ndarray:
    """
    Scalar complexity score (0=simplest, 1=most complex) for each GT case.

    Four normalised components (each → [0,1]), weighted sum:
      w=0.30  tip lateral displacement    ||p_tip[:2]||₂
      w=0.25  maximum lateral deflection  max_s ||p(s)[:2]||₂
      w=0.25  total curvature proxy       Σ arccos(t_i · t_{i+1})
      w=0.20  3-D nonplanarity            σ_min / (σ_sum + ε)
                  from SVD of the centered backbone point cloud

    The curvature proxy ≈ ∫|κ(s)|ds (discrete turning-angle sum).
    The nonplanarity term captures how much the backbone leaves the best-fit
    plane, distinguishing 3-D coils from in-plane bends.
    """
    def _norm01(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)

    # tip lateral displacement (xy)
    tip_lat = np.linalg.norm(positions_gt[:, -1, :2], axis=1)          # (N,)
    # max lateral deflection along arc
    max_lat = np.linalg.norm(positions_gt[:, :, :2], axis=2).max(1)    # (N,)
    # curvature proxy: sum of inter-segment turning angles
    dp       = np.diff(positions_gt, axis=1)                            # (N,M-1,3)
    seg_norm = np.linalg.norm(dp, axis=2, keepdims=True)
    dp_unit  = dp / np.where(seg_norm < 1e-12, 1.0, seg_norm)
    cos_ang  = np.clip((dp_unit[:, :-1] * dp_unit[:, 1:]).sum(2), -1.0, 1.0)
    curv_prx = np.degrees(np.arccos(cos_ang)).sum(1)                    # (N,)
    # 3-D nonplanarity: fraction of variance in minor SVD direction
    pts_c    = positions_gt - positions_gt.mean(axis=1, keepdims=True)  # (N,M,3)
    _, sv, _ = np.linalg.svd(pts_c, full_matrices=False)               # sv: (N,3)
    nonplan  = sv[:, 2] / (sv.sum(axis=1) + 1e-12)                     # (N,)

    return (0.30 * _norm01(tip_lat)
            + 0.25 * _norm01(max_lat)
            + 0.25 * _norm01(curv_prx)
            + 0.20 * _norm01(nonplan))


def _per_case_error_map(per_case: List[dict]) -> Dict[Tuple[int, str], float]:
    """Build (case_id, layout_name) → mean_centerline_error lookup."""
    acc: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for row in per_case:
        acc[(row["case_id"], row["layout_name"])].append(row["mean_centerline_error"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def select_cases_by_complexity_groups(
    complexity: np.ndarray,
    case_ids_gt: np.ndarray,
    layout_names: List[str],
    err_map: Dict,
    n_per_group: int = 2,
) -> List[Tuple[int, str, float]]:
    """
    Select n_per_group cases from each of three complexity terciles
    (low / mid / high).  Within each group, prefer cases where the two
    layouts differ most in mean-centerline-error (high inter-layout diversity).

    Returns list of (case_id, group_label, complexity_score) tuples ordered
    as: [low-A, low-B, mid-A, mid-B, high-A, high-B].
    """
    N     = len(complexity)
    order = np.argsort(complexity)                   # ascending index into GT arrays
    b     = [0, N // 3, 2 * N // 3, N]              # tercile boundaries
    g_names = ["low", "mid", "high"]
    g_labels = {"low": ["low-A", "low-B"],
                "mid": ["mid-A", "mid-B"],
                "high": ["high-A", "high-B"]}

    selected: List[Tuple[int, str, float]] = []
    for gi, gname in enumerate(g_names):
        cands = order[b[gi]: b[gi + 1]]             # indices into case_ids_gt
        # diversity = |err_layout0 − err_layout1|
        if len(layout_names) >= 2:
            divs = []
            for idx in cands:
                cid = int(case_ids_gt[idx])
                e0  = err_map.get((cid, layout_names[0]),  0.0)
                e1  = err_map.get((cid, layout_names[-1]), 0.0)
                divs.append(abs(e0 - e1))
            top = np.argsort(divs)[::-1]            # sort by diversity desc
            picks = [cands[top[j]] for j in range(min(n_per_group, len(cands)))]
        else:
            picks = list(cands[:n_per_group])
        # sort picks within group by complexity ascending (consistent display)
        picks = sorted(picks, key=lambda idx: complexity[idx])
        for j, idx in enumerate(picks):
            cid = int(case_ids_gt[idx])
            selected.append((cid, g_labels[gname][j], float(complexity[idx])))

    return selected   # [low-A, low-B, mid-A, mid-B, high-A, high-B]


# ---------------------------------------------------------------------------
# C  Frame and error-curve helpers
# ---------------------------------------------------------------------------

def _draw_frame_3d(
    ax,
    origin: np.ndarray,
    R: np.ndarray,
    scale: float = FRAME_SCALE,
    lw: float = FRAME_LW,
    alpha: float = 1.0,
) -> None:
    """Draw a small RGB reference frame triad at origin with rotation R."""
    for j, col in enumerate(_FRAME_AXIS_COLS):
        d = R[:, j] * scale
        ax.quiver(origin[0], origin[1], origin[2],
                  d[0], d[1], d[2],
                  color=col, linewidth=lw, alpha=alpha,
                  arrow_length_ratio=0.30, normalize=False)


def _plot_curve_colored_3d(
    ax,
    pts: np.ndarray,
    values: np.ndarray,
    norm: mcolors.Normalize,
    cmap,
    lw: float = 1.8,
    alpha: float = 0.95,
) -> Line3DCollection:
    """
    Draw a 3-D curve pts (M,3) with each segment coloured by the midpoint
    of values (M,) using the supplied norm and colormap.
    """
    seg_vals = 0.5 * (values[:-1] + values[1:])
    colors   = cmap(norm(seg_vals))
    segs     = [pts[[i, i + 1]] for i in range(len(pts) - 1)]   # list of (2,3)
    lc       = Line3DCollection(segs, colors=colors, linewidth=lw, alpha=alpha)
    ax.add_collection3d(lc)
    return lc


# ---------------------------------------------------------------------------
# C  Complexity-based overlay figure  (2 × 3 grid)
# ---------------------------------------------------------------------------

def plot_complexity_overlays(
    positions_gt: np.ndarray,
    orientations_gt: np.ndarray,
    case_ids_gt: np.ndarray,
    layout_names: List[str],
    shapes: Dict[str, dict],
    per_case: List[dict],
    n_cases: int = 6,
    elev: float = 25.0,
    azim: float = -60.0,
    frame_s_locs: Tuple[float, ...] = FRAME_S_LOCS,
    frame_scale: float = FRAME_SCALE,
    save_stem: Optional[Path] = None,
) -> None:
    """
    2 × 3 thesis-style overlay figure, cases selected by shape complexity.

    Layout (columns = complexity level):
      Row 0:  low-A  |  mid-A  |  high-A
      Row 1:  low-B  |  mid-B  |  high-B

    Curves
    ------
      GT backbone     — black solid  + small RGB frames at frame_s_locs
      last layout     — coloured by local position error (Line3DCollection)
                        + lighter RGB frames at same locations (if R_est saved)
      other layouts   — plain dashed line  + tip frame only (if R_est saved)

    Error colouring uses a global [0, max_err] scale across all 6 panels;
    one shared colourbar is placed on the right side of the figure.

    Panel annotations: case id, complexity group, per-layout mean error.
    """
    complexity = compute_shape_complexity_scores(positions_gt)
    err_map    = _per_case_error_map(per_case)
    selected   = select_cases_by_complexity_groups(
        complexity, case_ids_gt, layout_names, err_map, n_per_group=2
    )
    # selected = [low-A, low-B, mid-A, mid-B, high-A, high-B]
    # display order for 2×3 grid (columns = low | mid | high):
    #   [low-A, mid-A, high-A, low-B, mid-B, high-B]
    disp_order = [0, 2, 4, 1, 3, 5]
    disp = [selected[i] for i in disp_order]

    cmap_gt    = _cid_map(case_ids_gt)
    M          = positions_gt.shape[1]
    lname_col  = layout_names[-1]             # coloured curve = last layout (3-IMU)
    has_R_est  = "R_est" in shapes.get(lname_col, {})

    # ── print selection summary ─────────────────────────────────────────────
    print(f"  Case selection (2+2+2 by complexity):")
    for cid, lbl, score in selected:
        e_strs = "  ".join(
            f"{ln}={err_map.get((cid,ln),float('nan'))*1e3:.2f}mm"
            for ln in layout_names
        )
        print(f"    {lbl:8s}  case {cid:3d}  complexity={score:.3f}  {e_strs}")
    print(f"  Frame s-locations: {frame_s_locs}  scale={frame_scale*1e3:.1f} mm")
    print(f"  R_est available: {has_R_est}")

    # ── precompute global error range for coloured curve ─────────────────────
    d_col = shapes[lname_col]
    all_local_mm: List[np.ndarray] = []
    for cid, _, _ in disp:
        mask = (d_col["case_ids"] == cid) & (d_col["noise_real"] == 0)
        if np.any(mask):
            p_e = d_col["p_est"][mask][0]
            p_g = positions_gt[cmap_gt[cid]]
            all_local_mm.append(np.linalg.norm(p_e - p_g, axis=1) * 1e3)
    global_max_mm = float(np.concatenate(all_local_mm).max()) if all_local_mm else 1.0
    err_norm      = mcolors.Normalize(vmin=0.0, vmax=global_max_mm)
    err_cmap      = plt.get_cmap(_ERROR_CMAP)

    # ── global axis limits ───────────────────────────────────────────────────
    all_pts: List[np.ndarray] = []
    for cid, _, _ in disp:
        all_pts.append(positions_gt[cmap_gt[cid]])
        for d in shapes.values():
            mask = (d["case_ids"] == cid) & (d["noise_real"] == 0)
            if np.any(mask):
                all_pts.append(d["p_est"][mask][0])
    all_arr = np.vstack(all_pts)
    lo, hi  = all_arr.min(0), all_arr.max(0)
    span    = hi - lo
    pad     = np.where(span > 0, span * 0.08, 2e-3)
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])
    zlim = (lo[2] - pad[2], hi[2] + pad[2])

    # ── build figure with GridSpec (3D axes + colourbar column) ─────────────
    n_rows, n_cols = 2, 3
    fig = plt.figure(figsize=(5.4 * n_cols + 1.0, 5.6 * n_rows))
    gs  = fig.add_gridspec(
        n_rows, n_cols + 1,
        width_ratios=[1, 1, 1, 0.045],
        hspace=0.08, wspace=0.08,
        left=0.03, right=0.92, top=0.91, bottom=0.11,
    )
    axes_3d: List = []
    for r in range(n_rows):
        for c in range(n_cols):
            axes_3d.append(fig.add_subplot(gs[r, c], projection="3d"))
    cbar_ax = fig.add_subplot(gs[:, n_cols])

    legend_handles: list = []
    legend_labels:  list = []
    legend_done = False

    for panel_i, (cid, grp_lbl, cx_score) in enumerate(disp):
        ax      = axes_3d[panel_i]
        gt_idx  = cmap_gt[cid]
        p_gt    = positions_gt[gt_idx]        # (M, 3)
        R_gt_k  = orientations_gt[gt_idx]     # (M, 3, 3)

        # ── GT backbone ──
        h_gt, = ax.plot(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2],
                        "k-", lw=1.6, label="GT (Kirchhoff)")
        if not legend_done:
            legend_handles.append(h_gt)
            legend_labels.append("GT (Kirchhoff)")

        # ── GT orientation frames ──
        for s_loc in frame_s_locs:
            idx_s = int(round(s_loc * (M - 1)))
            _draw_frame_3d(ax, p_gt[idx_s], R_gt_k[idx_s],
                           scale=frame_scale, lw=FRAME_LW, alpha=0.90)

        ann_parts = [f"case {cid}  [{grp_lbl}]",
                     f"complexity  {cx_score:.3f}"]

        # ── estimated curves ──
        for li, lname in enumerate(layout_names):
            d    = shapes[lname]
            mask = (d["case_ids"] == cid) & (d["noise_real"] == 0)
            if not np.any(mask):
                continue
            p_est = d["p_est"][mask][0]         # (M, 3)
            st    = _lstyle(lname, layout_names)
            local_mm = np.linalg.norm(p_est - p_gt, axis=1) * 1e3

            if lname == lname_col:
                # last layout → colour by position error
                _plot_curve_colored_3d(ax, p_est, local_mm,
                                       err_norm, err_cmap,
                                       lw=st["lw"], alpha=0.95)
                if not legend_done:
                    proxy = plt.Line2D([0], [0], color=err_cmap(0.7),
                                       ls="-", lw=st["lw"],
                                       label=f"{lname} (err-coloured)")
                    legend_handles.append(proxy)
                    legend_labels.append(f"{lname} (err-coloured)")

                # estimated frames at all frame_s_locs (if R_est saved)
                if has_R_est:
                    R_est_k = d["R_est"][mask][0]
                    for s_loc in frame_s_locs:
                        idx_s = int(round(s_loc * (M - 1)))
                        _draw_frame_3d(ax, p_est[idx_s], R_est_k[idx_s],
                                       scale=frame_scale * 0.80,
                                       lw=FRAME_LW * 0.65, alpha=0.28)
            else:
                # other layouts → plain dashed line
                h_est, = ax.plot(p_est[:, 0], p_est[:, 1], p_est[:, 2],
                                 color=st["color"], ls=st["ls"], lw=st["lw"],
                                 alpha=0.88, label=lname)
                if not legend_done:
                    legend_handles.append(h_est)
                    legend_labels.append(lname)

                # tip frame only (if R_est saved, to avoid crowding)
                if has_R_est and "R_est" in d:
                    R_est_k = d["R_est"][mask][0]
                    idx_tip = int(round(1.0 * (M - 1)))
                    _draw_frame_3d(ax, p_est[idx_tip], R_est_k[idx_tip],
                                   scale=frame_scale * 0.75,
                                   lw=FRAME_LW * 0.55, alpha=0.22)

            err_mean_mm = err_map.get((cid, lname), float("nan")) * 1e3
            ann_parts.append(f"{lname}: {err_mean_mm:.2f} mm")

        legend_done = True

        # ── axis formatting ──
        ax.set_xlim(*xlim);  ax.set_ylim(*ylim);  ax.set_zlim(*zlim)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x [m]", fontsize=9)
        ax.set_ylabel("y [m]", fontsize=9)
        ax.set_zlabel("z [m]", fontsize=9)
        ax.tick_params(labelsize=8)

        ax.text2D(
            0.03, 0.97, "\n".join(ann_parts),
            transform=ax.transAxes,
            fontsize=ANN_FS, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2),
        )

    # ── shared colourbar ──
    sm = plt.cm.ScalarMappable(cmap=err_cmap, norm=err_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label(f"Pos. error — {lname_col}  [mm]", fontsize=LABEL_FS - 1)
    cb.ax.tick_params(labelsize=TICK_FS - 1)

    # ── legend ──
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", ncol=len(legend_handles),
        fontsize=LEGEND_FS - 1,
        bbox_to_anchor=(0.45, 0.01),
        framealpha=0.9,
    )
    fig.suptitle(
        "Shape overlays — GT vs EKF (cross-model), cases ranked by shape complexity",
        fontsize=TITLE_FS, y=0.97,
    )
    plt.show()
    if save_stem is not None:
        for ext in ("png", "pdf"):
            p = save_stem.parent / (save_stem.name + f"_overlays_complexity.{ext}")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Figure saved → {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Existing plots (unchanged)
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "mean_centerline_error"     : "Mean centerline error [mm]",
    "max_centerline_error"      : "Max centerline error [mm]",
    "tip_position_error"        : "Tip position error [mm]",
    "tip_orientation_error_deg" : "Tip orientation error [deg]",
    "mean_orientation_error_deg": "Mean orientation error [deg]",
}
MM_METRICS = {"mean_centerline_error", "max_centerline_error", "tip_position_error"}


def plot_cross_model_summary(
    summary: Dict[str, dict],
    layout_names: List[str],
    save_stem: Optional[Path] = None,
) -> None:
    """Two-panel bar chart: mean centerline error and tip position error."""
    n      = len(layout_names)
    x      = np.arange(n)
    colors = [_lstyle(nm, layout_names)["color"] for nm in layout_names]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    panels = [
        ("mean_centerline_error", "Mean centerline error  [m]"),
        ("tip_position_error",    "Tip position error  [m]"),
    ]
    for ax, (metric, ylabel) in zip(axes, panels):
        means = [summary[nm][metric + "_mean"] for nm in layout_names]
        stds  = [summary[nm][metric + "_std"]  for nm in layout_names]
        bars  = ax.bar(x, means, yerr=stds, color=colors, capsize=5,
                       width=0.5, alpha=0.85, error_kw=dict(lw=1.2))
        ax.set_xticks(x)
        ax.set_xticklabels(layout_names, rotation=12, ha="right",
                           fontsize=TICK_FS)
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{m*1e3:.2f} mm",
                    ha="center", va="bottom", fontsize=9)

    fig.suptitle("Cross-Model Robustness: Kirchhoff-rod shape estimation",
                 fontsize=TITLE_FS)
    _save_show(fig, save_stem, "summary")


def plot_metric_distributions(
    per_case: List[dict],
    layout_names: List[str],
    save_stem: Optional[Path] = None,
) -> None:
    """Violin plot of each geometry metric, grouped by layout."""
    groups: Dict[str, Dict[str, List[float]]] = {
        nm: defaultdict(list) for nm in layout_names
    }
    for row in per_case:
        lname = row["layout_name"]
        if lname not in groups:
            continue
        for mk in METRIC_LABELS:
            val = row[mk] * (1e3 if mk in MM_METRICS else 1.0)
            groups[lname][mk].append(val)

    n_metrics  = len(METRIC_LABELS)
    metric_keys = list(METRIC_LABELS.keys())
    fig, axes  = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4.5))

    for ax, mk in zip(axes, metric_keys):
        data_per = [groups[nm][mk] for nm in layout_names]
        parts    = ax.violinplot(data_per, showmedians=True, showextrema=True)
        for pc, nm in zip(parts["bodies"], layout_names):
            pc.set_facecolor(_lstyle(nm, layout_names)["color"])
            pc.set_alpha(0.70)
        for comp in ("cmedians", "cmins", "cmaxes", "cbars"):
            if comp in parts:
                parts[comp].set_color("k")
                parts[comp].set_linewidth(0.9)
        ax.set_xticks(range(1, len(layout_names) + 1))
        ax.set_xticklabels(layout_names, rotation=12, ha="right", fontsize=TICK_FS)
        ax.set_ylabel(METRIC_LABELS[mk], fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle("Metric distributions — cross-model EKF (Kirchhoff rod)",
                 fontsize=TITLE_FS)
    _save_show(fig, save_stem, "metrics")


def plot_representative_overlays(
    positions_gt: np.ndarray,
    case_ids_gt: np.ndarray,
    layout_names: List[str],
    shapes: Dict[str, dict],
    n_cases: int = 3,
    save_stem: Optional[Path] = None,
) -> None:
    """Legacy: low/median/high error overlays (error ranked from first layout)."""
    cmap_gt = _cid_map(case_ids_gt)
    lname0  = layout_names[0]
    d0      = shapes[lname0]
    mask0   = d0["noise_real"] == 0
    c0      = d0["case_ids"][mask0]
    p0      = d0["p_est"][mask0]

    case_err: Dict[int, float] = {}
    for k, cid in enumerate(c0):
        gt_idx = cmap_gt.get(int(cid))
        if gt_idx is None:
            continue
        p_gt = positions_gt[gt_idx]
        case_err[int(cid)] = float(np.mean(np.linalg.norm(p0[k] - p_gt, axis=1)))

    sorted_cases = sorted(case_err.items(), key=lambda x: x[1])
    n_total  = len(sorted_cases)
    picks    = [0, n_total // 2, n_total - 1][:n_cases]
    rep_cases = [(sorted_cases[i][0], lbl) for i, lbl in
                 zip(picks, ["low error", "median error", "high error"][:n_cases])]

    fig = plt.figure(figsize=(5 * n_cases, 5))
    for col, (cid, lbl) in enumerate(rep_cases):
        ax     = fig.add_subplot(1, n_cases, col + 1, projection="3d")
        gt_idx = cmap_gt[cid]
        p_gt   = positions_gt[gt_idx]
        ax.plot(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2],
                "k-", lw=1.8, label="GT (Kirchhoff)")
        all_pos = [p_gt]
        for lname in layout_names:
            d    = shapes[lname]
            mask = (d["case_ids"] == cid) & (d["noise_real"] == 0)
            if not np.any(mask):
                continue
            p_est = d["p_est"][mask][0]
            st    = _lstyle(lname, layout_names)
            ax.plot(p_est[:, 0], p_est[:, 1], p_est[:, 2],
                    color=st["color"], ls=st["ls"], lw=st["lw"],
                    alpha=0.9, label=lname)
            all_pos.append(p_est)
        ax.set_title(f"case {cid}  ({lbl})", fontsize=9)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.set_zlabel("z [m]", fontsize=8)
        ax.set_box_aspect([1, 1, 1])
        _tight_axis_limits(ax, all_pos)
        if col == 0:
            ax.legend(fontsize=LEGEND_FS - 2, loc="upper left")

    fig.suptitle("Shape overlays — GT vs EKF estimate (cross-model)", fontsize=TITLE_FS)
    _save_show(fig, save_stem, "overlays")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot saved Kirchhoff-rod shape estimation results "
                    "(no EKF re-run).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", default="gt_data/results",
        help="directory containing kirchhoff_shape_est_*.json / *_shapes.npz",
    )
    parser.add_argument("--json",   default=None,
                        help="explicit path to kirchhoff_shape_est_results.json")
    parser.add_argument("--shapes", default=None,
                        help="explicit path to kirchhoff_shape_est_shapes.npz")
    parser.add_argument("--gt",     default="gt_data/kirchhoff_gt_dataset.npz",
                        help="Step-1 ground-truth NPZ")

    # which figures to generate
    parser.add_argument("--plot-summary",            action="store_true", default=True)
    parser.add_argument("--no-plot-summary",         dest="plot_summary",
                        action="store_false")
    parser.add_argument("--plot-metrics",            action="store_true", default=False)
    parser.add_argument("--plot-overlays",           action="store_true", default=False)
    parser.add_argument("--plot-arclength-pos",      action="store_true", default=False)
    parser.add_argument("--plot-arclength-ori",      action="store_true", default=False)
    parser.add_argument("--plot-complexity-overlays",action="store_true", default=False)
    parser.add_argument("--plot-all",                action="store_true", default=False,
                        help="enable all figure types")

    # overlay options
    parser.add_argument("--num-overlay-cases",       type=int,   default=3)
    parser.add_argument("--elev",                    type=float, default=25.0,
                        help="3-D view elevation for complexity overlays")
    parser.add_argument("--azim",                    type=float, default=-60.0,
                        help="3-D view azimuth for complexity overlays")

    parser.add_argument("--no-save", action="store_true",
                        help="display figures but do not write PNG/PDF")
    args = parser.parse_args()

    # expand --plot-all
    if args.plot_all:
        args.plot_summary             = True
        args.plot_metrics             = True
        args.plot_overlays            = True
        args.plot_arclength_pos       = True
        args.plot_arclength_ori       = True
        args.plot_complexity_overlays = True

    results_dir  = Path(args.results_dir)
    json_path    = Path(args.json)   if args.json   else results_dir / "kirchhoff_shape_est_results.json"
    shapes_path  = Path(args.shapes) if args.shapes else results_dir / "kirchhoff_shape_est_shapes.npz"
    gt_path      = Path(args.gt)
    save_stem    = None if args.no_save else (results_dir / "kirchhoff_shape_est")

    # ------------------------------------------------------------------ load JSON
    if not json_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {json_path}")
    data_json    = load_json(json_path)
    summary      = data_json["summary"]
    per_case     = data_json["per_case"]
    layout_names = list(summary.keys())
    print(f"Loaded {len(per_case)} rows from {json_path}")
    print(f"Layouts: {layout_names}")

    # ------------------------------------------------------------------ GT
    need_gt = (args.plot_arclength_pos or args.plot_arclength_ori
               or args.plot_complexity_overlays or args.plot_overlays)
    positions_gt: Optional[np.ndarray] = None
    orientations_gt: Optional[np.ndarray] = None
    case_ids_gt: Optional[np.ndarray] = None
    if need_gt:
        if not gt_path.exists():
            raise FileNotFoundError(f"GT NPZ not found: {gt_path}")
        positions_gt, orientations_gt, case_ids_gt = load_gt(gt_path)
        print(f"Loaded {len(positions_gt)} GT cases from {gt_path}")

    # ------------------------------------------------------------------ shapes
    need_shapes = (args.plot_arclength_pos or args.plot_arclength_ori
                   or args.plot_complexity_overlays or args.plot_overlays)
    shapes: Optional[Dict[str, dict]] = None
    if need_shapes:
        if not shapes_path.exists():
            raise FileNotFoundError(
                f"Shapes NPZ not found: {shapes_path}\n"
                "Re-run evaluate_kirchhoff_shape_estimation.py to generate it."
            )
        _, shapes = load_shapes(shapes_path)
        M       = next(iter(shapes.values()))["p_est"].shape[1]
        s_grid  = np.linspace(0.0, 1.0, M)
        print(f"Loaded estimated shapes (M={M} arc-length points)")

    # ------------------------------------------------------------------ figures

    if args.plot_summary:
        print("\n--- summary bar chart ---")
        plot_cross_model_summary(summary, layout_names, save_stem)

    if args.plot_metrics:
        print("\n--- metric distributions ---")
        plot_metric_distributions(per_case, layout_names, save_stem)

    if args.plot_arclength_pos:
        print("\n--- arc-length position error ---")
        pos_errors = compute_arclength_position_errors(
            shapes, positions_gt, case_ids_gt)
        plot_arclength_position_errors(pos_errors, s_grid, layout_names, save_stem)

    if args.plot_arclength_ori:
        print("\n--- arc-length orientation (tangent direction) error ---")
        ori_errors = compute_arclength_orientation_errors(
            shapes, orientations_gt, case_ids_gt)
        plot_arclength_orientation_errors(ori_errors, s_grid, layout_names, save_stem)

    if args.plot_complexity_overlays:
        print("\n--- complexity-based shape overlays ---")
        plot_complexity_overlays(
            positions_gt  = positions_gt,
            orientations_gt = orientations_gt,
            case_ids_gt   = case_ids_gt,
            layout_names  = layout_names,
            shapes        = shapes,
            per_case      = per_case,
            n_cases       = 6,
            elev          = args.elev,
            azim          = args.azim,
            save_stem     = save_stem,
        )

    if args.plot_overlays:
        print("\n--- legacy error-ranked overlays ---")
        plot_representative_overlays(
            positions_gt  = positions_gt,
            case_ids_gt   = case_ids_gt,
            layout_names  = layout_names,
            shapes        = shapes,
            n_cases       = args.num_overlay_cases,
            save_stem     = save_stem,
        )


if __name__ == "__main__":
    main()
