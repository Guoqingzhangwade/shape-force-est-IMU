#!/usr/bin/env python3
"""
Plot-only script for the Cross-Model Robustness Study.
Generates TWO separate output figures; EKF is NOT re-run.

Figure 1  —  Main aggregate (2×2, all 50 cases × 5 realisations)
-----------------------------------------------------------------
(a) grouped violins: mean centerline error + tip position error  [mm]
(b) grouped violins: mean orientation error + tip orientation error  [deg]
(c) arc-length position error: mean + IQR band over all cases  [mm]
(d) arc-length SO(3) orientation error: mean + IQR band  [deg]

Intended for: main manuscript text.

Figure 2  —  Representative-case detail (3 cols × 3 rows, 3 cases)
-------------------------------------------------------------------
Row 0: 3-D shape overlays (GT + per-layout EKF estimate, noise_real=0)
Row 1: per-case arc-length position error (mean + IQR, all 5 realisations)
Row 2: per-case arc-length SO(3) orientation error (mean + IQR)

Cases selected by shape complexity (low / mid / high), preferring the
case within each group where the two layouts differ most in performance.

Intended for: appendix / thesis supplementary detail.

Required files
--------------
  gt_data/results/kirchhoff_shape_est_results.json
  gt_data/results/kirchhoff_shape_est_shapes.npz
  gt_data/kirchhoff_gt_dataset.npz

Outputs
-------
  kirchhoff_shape_est_main_aggregate_violin.png / .pdf / _transparent.png
  kirchhoff_shape_est_composite_nonplanar_clean.png / .pdf

Usage
-----
  python plot_kirchhoff_shape_est_composite.py           # both figures
  python plot_kirchhoff_shape_est_composite.py --plot-main-aggregate
  python plot_kirchhoff_shape_est_composite.py --plot-representative
  python plot_kirchhoff_shape_est_composite.py --no-save
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
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

LABEL_FS  = 12
TICK_FS   = 11
TITLE_FS  = 11
PANEL_FS  = 13      # bold (a) (b) panel labels
LEGEND_FS = 11
ANN_FS    = 7.5     # 3-D panel annotation text

COLORS = {"2-IMU": "#1f77b4", "3-IMU": "#d62728"}   # blue / red
LS     = {"2-IMU": "--",       "3-IMU": "-."}
LW     = 1.8

# grouped-violin geometry
_GROUP_CENTRES = [1.0, 3.0]
_HALF_GAP      = 0.22
_VWIDTH        = 0.36

# IMU arc-length hints for vertical markers
_IMU_S = {"2-IMU": [0.50, 1.00], "3-IMU": [0.25, 0.50, 1.00]}

# default 3-D view angle
_ELEV, _AZIM = 25.0, -60.0


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

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_gt(gt_npz: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns positions (N,M,3), orientations (N,M,3,3), case_ids (N,)."""
    data = np.load(gt_npz, allow_pickle=True)
    N, M, _ = data["positions"].shape
    ori = data["orientations"].reshape(N, M, 3, 3)
    return data["positions"], ori, data["case_id"]


def _load_shapes(shapes_npz: Path) -> Tuple[List[str], Dict[str, dict]]:
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


def _cid_map(case_ids_gt: np.ndarray) -> Dict[int, int]:
    return {int(cid): i for i, cid in enumerate(case_ids_gt)}


# ---------------------------------------------------------------------------
# Arc-length error computation  (global: all rows)
# ---------------------------------------------------------------------------

def compute_arclength_position_errors(
    shapes: Dict[str, dict],
    positions_gt: np.ndarray,
    case_ids_gt: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Returns {layout: (n_rows, M)} position error in metres."""
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
    """Returns {layout: (n_rows, M)} SO(3) error in degrees. Requires R_est."""
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
            R_est_k = d["R_est"][k]
            R_gt_k  = orientations_gt[gt_idx]
            for i in range(M):
                errs[k, i] = np.degrees(
                    np.linalg.norm(_so3_log(R_est_k[i] @ R_gt_k[i].T))
                )
        out[lname] = errs
    return out


# ---------------------------------------------------------------------------
# Complexity helpers  (used by Figure 2 case selection)
# ---------------------------------------------------------------------------

def _compute_complexity(positions_gt: np.ndarray) -> np.ndarray:
    """
    Weighted complexity score in [0,1] per GT case.

    Components (each normalised to [0,1]):
      0.30  tip lateral displacement  ||p_tip[:2]||
      0.25  max lateral deflection    max_s ||p(s)[:2]||
      0.25  total curvature proxy     Σ arccos(t_i·t_{i+1})
      0.20  3-D nonplanarity          σ_min / σ_sum  (SVD of centred backbone)
    """
    def _n01(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)

    tip_lat = np.linalg.norm(positions_gt[:, -1, :2], axis=1)
    max_lat = np.linalg.norm(positions_gt[:, :, :2], axis=2).max(1)
    dp      = np.diff(positions_gt, axis=1)
    dp_u    = dp / np.where(np.linalg.norm(dp, axis=2, keepdims=True) < 1e-12,
                            1.0, np.linalg.norm(dp, axis=2, keepdims=True))
    cos_a   = np.clip((dp_u[:, :-1] * dp_u[:, 1:]).sum(2), -1.0, 1.0)
    curv    = np.degrees(np.arccos(cos_a)).sum(1)
    pts_c   = positions_gt - positions_gt.mean(axis=1, keepdims=True)
    _, sv, _ = np.linalg.svd(pts_c, full_matrices=False)
    nonplan = sv[:, 2] / (sv.sum(axis=1) + 1e-12)

    return (0.30 * _n01(tip_lat) + 0.25 * _n01(max_lat)
            + 0.25 * _n01(curv)  + 0.20 * _n01(nonplan))


def _per_case_mean_errors(per_case: List[dict]) -> Dict[Tuple[int, str], float]:
    """Build (case_id, layout) → mean_centerline_error lookup."""
    acc: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for row in per_case:
        acc[(row["case_id"], row["layout_name"])].append(row["mean_centerline_error"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _select_3_cases(
    complexity: np.ndarray,
    case_ids_gt: np.ndarray,
    layout_names: List[str],
    err_map: Dict,
) -> List[Tuple[int, str, float]]:
    """
    Pick one case per complexity tercile (low / mid / high).
    Within each tercile prefer the case with largest inter-layout error gap.
    Returns [(case_id, label, complexity_score), ...].
    """
    N     = len(complexity)
    order = np.argsort(complexity)
    b     = [0, N // 3, 2 * N // 3, N]
    out   = []
    for gi, lbl in enumerate(("low", "mid", "high")):
        cands = order[b[gi]: b[gi + 1]]
        if len(layout_names) >= 2:
            divs = []
            for idx in cands:
                cid = int(case_ids_gt[idx])
                e0  = err_map.get((cid, layout_names[0]),  0.0)
                e1  = err_map.get((cid, layout_names[-1]), 0.0)
                divs.append(abs(e0 - e1))
            best = cands[int(np.argmax(divs))]
        else:
            best = cands[len(cands) // 2]
        cid = int(case_ids_gt[best])
        out.append((cid, lbl, float(complexity[best])))
    return out


# ---------------------------------------------------------------------------
# Per-case arc-length errors  (for Figure 2 rows 1 & 2)
# ---------------------------------------------------------------------------

def _case_arclength_errors(
    cid: int,
    shapes: Dict[str, dict],
    positions_gt: np.ndarray,
    orientations_gt: np.ndarray,
    cmap_gt: Dict[int, int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    For a single case_id, return per-(noise_real, arc_pt) errors across all
    noise realisations.

    Returns
    -------
    pos_err  : {layout: (n_noise, M)}  position error [m]
    ori_err  : {layout: (n_noise, M)}  SO(3) error [deg]  (only if R_est saved)
    """
    gt_idx = cmap_gt[cid]
    p_gt   = positions_gt[gt_idx]       # (M, 3)
    R_gt   = orientations_gt[gt_idx]    # (M, 3, 3)

    pos_err: Dict[str, np.ndarray] = {}
    ori_err: Dict[str, np.ndarray] = {}

    for lname, d in shapes.items():
        mask  = d["case_ids"] == cid    # all noise realisations
        p_est = d["p_est"][mask]        # (n_noise, M, 3)
        pos_err[lname] = np.linalg.norm(p_est - p_gt[None], axis=2)  # (n_noise, M)

        if "R_est" in d:
            R_est  = d["R_est"][mask]   # (n_noise, M, 3, 3)
            n_n, M_  = R_est.shape[:2]
            o = np.zeros((n_n, M_))
            for r in range(n_n):
                for i in range(M_):
                    o[r, i] = np.degrees(
                        np.linalg.norm(_so3_log(R_est[r, i] @ R_gt[i].T))
                    )
            ori_err[lname] = o

    return pos_err, ori_err


# ---------------------------------------------------------------------------
# Panel helpers  (shared by both figures)
# ---------------------------------------------------------------------------

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

            parts = ax.violinplot(
                [vals], positions=[xpos], widths=_VWIDTH,
                showmeans=False, showmedians=False, showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(COLORS[lname])
                body.set_edgecolor("none")
                body.set_alpha(0.60)

            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            ax.vlines(xpos, q25, q75, color=COLORS[lname], linewidth=3.2, zorder=5)
            ax.scatter([xpos], [med], color="white", s=28, zorder=6,
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
    """Global mean + IQR band over all rows."""
    for lname in layout_names:
        e    = errors[lname] * (1e3 if scale_mm else 1.0)
        mean = np.nanmean(e,           axis=0)
        p25  = np.nanpercentile(e, 25, axis=0)
        p75  = np.nanpercentile(e, 75, axis=0)
        col  = COLORS[lname]
        ax.plot(s_grid, mean, color=col, lw=LW)
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
    """Per-case mean + IQR band (n_noise realisations)."""
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
        ax.set_xlabel("$s$", fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS - 1)
    ax.set_xlim(s_grid[0], s_grid[-1])
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, alpha=0.30, linestyle="--")
    ax.set_axisbelow(True)


def _add_imu_markers(ax, layout_names: List[str]) -> None:
    done: set = set()
    for lname in layout_names:
        for sp in _IMU_S.get(lname, []):
            if sp not in done:
                ax.axvline(sp, color="grey", lw=0.7, ls=":", alpha=0.55, zorder=0)
                done.add(sp)


# ---------------------------------------------------------------------------
# Figure 1 — Main aggregate  (2 × 2)
# ---------------------------------------------------------------------------

def make_aggregate_figure(
    per_case: List[dict],
    layout_names: List[str],
    pos_errors: Dict[str, np.ndarray],
    ori_errors: Dict[str, np.ndarray],
    s_grid: np.ndarray,
    n_cases: int,
    n_noise: int,
    save_stem: Optional[Path] = None,
) -> None:
    """2×2 summary figure covering all cases and noise realisations."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    _panel_violin(ax_a, layout_names,
                  ["mean_centerline_error", "tip_position_error"],
                  per_case, scale_mm=True, ylabel="Error  [mm]",
                  metric_xlabels=["Mean centerline\nerror", "Tip position\nerror"])
    ax_a.set_title("Position metrics", fontsize=TITLE_FS)

    _panel_violin(ax_b, layout_names,
                  ["mean_orientation_error_deg", "tip_orientation_error_deg"],
                  per_case, scale_mm=False, ylabel="Error  [deg]",
                  metric_xlabels=["Mean orientation\nerror", "Tip orientation\nerror"])
    ax_b.set_title("Orientation metrics", fontsize=TITLE_FS)

    _panel_arclength_global(ax_c, pos_errors, s_grid, layout_names,
                            ylabel="Position error  [mm]", scale_mm=True)
    ax_c.set_title("Position error along arc-length", fontsize=TITLE_FS)

    _panel_arclength_global(ax_d, ori_errors, s_grid, layout_names,
                            ylabel="Orientation error  [deg]")
    ax_d.set_title("Orientation error along arc-length", fontsize=TITLE_FS)

    for ax, lbl in zip([ax_a, ax_b, ax_c, ax_d],
                       ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(0.02, 0.98, lbl, transform=ax.transAxes,
                fontsize=PANEL_FS, fontweight="bold", va="top", ha="left")

    proxies = [mpatches.Patch(facecolor=COLORS[nm], alpha=0.75, label=nm)
               for nm in layout_names]
    fig.legend(handles=proxies, loc="lower center", ncol=len(layout_names),
               fontsize=LEGEND_FS, framealpha=0.9, bbox_to_anchor=(0.5, 0.00))

    ax_d.text(0.98, 0.97,
              f"n = {n_cases} cases × {n_noise} realisations",
              transform=ax_d.transAxes, fontsize=8,
              ha="right", va="top", color="grey", style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

    if save_stem is not None:
        for ext in ("png", "pdf"):
            p = save_stem.parent / (save_stem.name + f".{ext}")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved → {p}")
        p_tr = save_stem.parent / (save_stem.name + "_transparent.png")
        fig.savefig(p_tr, dpi=150, bbox_inches="tight", transparent=True)
        print(f"Saved → {p_tr}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Representative cases  (3 cols × 3 rows)
# ---------------------------------------------------------------------------

def make_representative_figure(
    selected: List[Tuple[int, str, float]],
    positions_gt: np.ndarray,
    orientations_gt: np.ndarray,
    case_ids_gt: np.ndarray,
    layout_names: List[str],
    shapes: Dict[str, dict],
    err_map: Dict,
    s_grid: np.ndarray,
    elev: float = _ELEV,
    azim: float = _AZIM,
    save_stem: Optional[Path] = None,
) -> None:
    """
    3 × 3 detail figure for three representative complexity cases.

    Row 0: 3-D shape overlays (GT + EKF, noise_real = 0)
    Row 1: per-case arc-length position error  [mm]
    Row 2: per-case arc-length SO(3) orientation error  [deg]
    """
    n_cases = len(selected)
    cmap_gt = _cid_map(case_ids_gt)
    M       = positions_gt.shape[1]

    # ── global 3-D axis limits (consistent across top row) ────────────────
    all_pts: List[np.ndarray] = []
    for cid, _, _ in selected:
        all_pts.append(positions_gt[cmap_gt[cid]])
        for d in shapes.values():
            mask = (d["case_ids"] == cid) & (d["noise_real"] == 0)
            if np.any(mask):
                all_pts.append(d["p_est"][mask][0])
    arr    = np.vstack(all_pts)
    lo, hi = arr.min(0), arr.max(0)
    span   = hi - lo
    pad    = np.where(span > 0, span * 0.08, 2e-3)
    xlim   = (lo[0] - pad[0], hi[0] + pad[0])
    ylim   = (lo[1] - pad[1], hi[1] + pad[1])
    zlim   = (lo[2] - pad[2], hi[2] + pad[2])

    # ── figure with mixed row heights ─────────────────────────────────────
    fig = plt.figure(figsize=(5.0 * n_cases, 11.5))
    gs  = fig.add_gridspec(
        3, n_cases,
        height_ratios=[4.2, 2.2, 2.2],
        hspace=0.30, wspace=0.22,
        left=0.07, right=0.97, top=0.94, bottom=0.08,
    )

    legend_handles: list = []
    legend_labels:  list = []
    legend_done = False

    for col, (cid, grp_lbl, cx_score) in enumerate(selected):
        gt_idx = cmap_gt[cid]
        p_gt   = positions_gt[gt_idx]

        # ── row 0: 3-D overlay ────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[0, col], projection="3d")

        h_gt, = ax3.plot(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2],
                         "k-", lw=1.6, label="GT (Kirchhoff)")
        if not legend_done:
            legend_handles.append(h_gt)
            legend_labels.append("GT (Kirchhoff)")

        ann_lines = [f"case {cid}  [{grp_lbl}]",
                     f"complexity {cx_score:.3f}"]

        for lname in layout_names:
            d    = shapes[lname]
            mask = (d["case_ids"] == cid) & (d["noise_real"] == 0)
            if not np.any(mask):
                continue
            p_est = d["p_est"][mask][0]
            h, = ax3.plot(p_est[:, 0], p_est[:, 1], p_est[:, 2],
                          color=COLORS[lname], ls=LS[lname], lw=LW,
                          alpha=0.90, label=lname)
            if not legend_done:
                legend_handles.append(h)
                legend_labels.append(lname)
            emm = err_map.get((cid, lname), float("nan")) * 1e3
            ann_lines.append(f"{lname}: {emm:.2f} mm")

        legend_done = True

        ax3.set_xlim(*xlim); ax3.set_ylim(*ylim); ax3.set_zlim(*zlim)
        ax3.set_box_aspect([1, 1, 1])
        ax3.view_init(elev=elev, azim=azim)
        ax3.set_xlabel("x [m]", fontsize=8)
        ax3.set_ylabel("y [m]", fontsize=8)
        ax3.set_zlabel("z [m]", fontsize=8)
        ax3.tick_params(labelsize=7)
        ax3.text2D(0.03, 0.97, "\n".join(ann_lines),
                   transform=ax3.transAxes, fontsize=ANN_FS,
                   va="top", ha="left",
                   bbox=dict(facecolor="white", alpha=0.65,
                             edgecolor="none", pad=2))

        # row 0 panel label
        ax3.set_title(f"({chr(ord('a') + col)})  {grp_lbl} complexity",
                      fontsize=TITLE_FS)

        # ── row 1: per-case position error ────────────────────────────────
        ax_p = fig.add_subplot(gs[1, col])
        pe, oe = _case_arclength_errors(
            cid, shapes, positions_gt, orientations_gt, cmap_gt
        )
        _panel_arclength_case(ax_p, pe, s_grid, layout_names,
                              ylabel="Pos. err. [mm]",
                              scale_mm=True, show_xlabel=False)

        # ── row 2: per-case orientation error ─────────────────────────────
        ax_o = fig.add_subplot(gs[2, col])
        _panel_arclength_case(ax_o, oe, s_grid, layout_names,
                              ylabel="Ori. err. [deg]",
                              scale_mm=False, show_xlabel=True)

    # ── shared legend below figure ────────────────────────────────────────
    proxies = [mpatches.Patch(facecolor=COLORS[nm], alpha=0.75, label=nm)
               for nm in layout_names]
    gt_proxy = plt.Line2D([0], [0], color="k", lw=1.6, label="GT (Kirchhoff)")
    fig.legend(
        handles=[gt_proxy] + proxies,
        loc="lower center", ncol=1 + len(layout_names),
        fontsize=LEGEND_FS, framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.show()

    if save_stem is not None:
        for ext in ("png", "pdf"):
            p = save_stem.parent / (save_stem.name + f".{ext}")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved → {p}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate both cross-model robustness figures (no EKF re-run).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default="gt_data/results")
    parser.add_argument("--gt", default="gt_data/kirchhoff_gt_dataset.npz")
    parser.add_argument("--plot-main-aggregate", action="store_true", default=False,
                        help="generate the 2×2 aggregate figure only")
    parser.add_argument("--plot-representative",  action="store_true", default=False,
                        help="generate the 3-case representative figure only")
    parser.add_argument("--elev", type=float, default=_ELEV,
                        help="3-D view elevation for representative figure")
    parser.add_argument("--azim", type=float, default=_AZIM,
                        help="3-D view azimuth for representative figure")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    # default: generate both figures
    if not args.plot_main_aggregate and not args.plot_representative:
        args.plot_main_aggregate = True
        args.plot_representative  = True

    results_dir = Path(args.results_dir)
    json_path   = results_dir / "kirchhoff_shape_est_results.json"
    shapes_path = results_dir / "kirchhoff_shape_est_shapes.npz"
    gt_path     = Path(args.gt)

    for p in (json_path, shapes_path, gt_path):
        if not p.exists():
            raise FileNotFoundError(p)

    # ── load shared data ───────────────────────────────────────────────────
    data_json    = _load_json(json_path)
    per_case     = data_json["per_case"]
    layout_names = list(data_json["summary"].keys())
    print(f"JSON: {len(per_case)} rows, layouts {layout_names}")

    positions_gt, orientations_gt, case_ids_gt = _load_gt(gt_path)
    n_cases = len(positions_gt)
    M       = positions_gt.shape[1]
    print(f"GT: {n_cases} cases, {M} arc-length pts")

    _, shapes = _load_shapes(shapes_path)
    n_noise = len(shapes[layout_names[0]]["p_est"]) // n_cases
    s_grid  = np.linspace(0.0, 1.0, M)
    print(f"Shapes NPZ: n_noise~{n_noise}, "
          f"R_est={'R_est' in shapes[layout_names[0]]}")

    # ── arc-length errors (needed by both figures) ─────────────────────────
    print("Computing arc-length position errors …")
    pos_errors = compute_arclength_position_errors(shapes, positions_gt, case_ids_gt)
    print("Computing arc-length SO(3) orientation errors …")
    ori_errors = compute_arclength_orientation_errors(
        shapes, orientations_gt, case_ids_gt
    )

    print(f"\nData scope: {n_cases} cases × {n_noise} realisations per layout")
    for lname in layout_names:
        me = np.nanmean(pos_errors[lname]) * 1e3
        oe = np.nanmean(ori_errors[lname])
        print(f"  {lname:<8s}  pos={me:.3f} mm   ori={oe:.3f} deg")

    # ── Figure 1: aggregate ────────────────────────────────────────────────
    if args.plot_main_aggregate:
        stem1 = (None if args.no_save
                 else results_dir / "kirchhoff_shape_est_main_aggregate_violin")
        print("\n--- Figure 1: 2×2 aggregate ---")
        make_aggregate_figure(
            per_case=per_case, layout_names=layout_names,
            pos_errors=pos_errors, ori_errors=ori_errors,
            s_grid=s_grid, n_cases=n_cases, n_noise=n_noise,
            save_stem=stem1,
        )

    # ── Figure 2: representative cases ────────────────────────────────────
    if args.plot_representative:
        stem2 = (None if args.no_save
                 else results_dir / "kirchhoff_shape_est_composite_nonplanar_clean")
        print("\n--- Figure 2: representative-case detail ---")
        complexity = _compute_complexity(positions_gt)
        err_map    = _per_case_mean_errors(per_case)
        selected   = _select_3_cases(complexity, case_ids_gt, layout_names, err_map)
        print("  Selected cases (low / mid / high complexity):")
        for cid, lbl, sc in selected:
            e_str = "  ".join(
                f"{ln}={err_map.get((cid,ln),float('nan'))*1e3:.2f}mm"
                for ln in layout_names
            )
            print(f"    {lbl:4s}  case {cid:3d}  cx={sc:.3f}  {e_str}")
        make_representative_figure(
            selected=selected,
            positions_gt=positions_gt,
            orientations_gt=orientations_gt,
            case_ids_gt=case_ids_gt,
            layout_names=layout_names,
            shapes=shapes,
            err_map=err_map,
            s_grid=s_grid,
            elev=args.elev,
            azim=args.azim,
            save_stem=stem2,
        )


if __name__ == "__main__":
    main()
