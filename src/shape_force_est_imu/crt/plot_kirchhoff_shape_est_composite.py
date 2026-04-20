#!/usr/bin/env python3
"""
Main-text aggregate figure for the Cross-Model Robustness Study.

Creates one 2×2 composite manuscript figure summarising all 50 Kirchhoff-rod
test cases and all 5 IMU-noise realisations (500 rows per layout).

Panels
------
(a) top-left  — violin: mean centerline error + tip position error  [mm]
(b) top-right — violin: mean orientation error + tip orientation error [deg]
(c) bot-left  — arc-length position error vs s  (mean + IQR band)  [mm]
(d) bot-right — arc-length SO(3) orientation error vs s             [deg]

All data read from already-saved files; EKF is NOT re-run.

Required files
--------------
  gt_data/results/kirchhoff_shape_est_results.json
  gt_data/results/kirchhoff_shape_est_shapes.npz
  gt_data/kirchhoff_gt_dataset.npz

Output
------
  gt_data/results/kirchhoff_shape_est_main_aggregate_violin.png
  gt_data/results/kirchhoff_shape_est_main_aggregate_violin.pdf
  gt_data/results/kirchhoff_shape_est_main_aggregate_violin_transparent.png

Usage
-----
  python plot_kirchhoff_shape_est_composite.py
  python plot_kirchhoff_shape_est_composite.py --results-dir gt_data/results
  python plot_kirchhoff_shape_est_composite.py --no-save
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

LABEL_FS = 12
TICK_FS  = 11
TITLE_FS = 11
PANEL_FS = 13      # bold (a) (b) labels
LEGEND_FS = 11

COLORS = {"2-IMU": "#1f77b4", "3-IMU": "#d62728"}   # blue / red

# grouped-violin geometry
_GROUP_CENTRES = [1.0, 3.0]   # x-centre for each metric group
_HALF_GAP      = 0.22         # half distance between the two layout violins
_VWIDTH        = 0.36         # body width

# heuristic IMU arc-length positions (for vertical marker lines in panels c/d)
_IMU_S = {"2-IMU": [0.50, 1.00], "3-IMU": [0.25, 0.50, 1.00]}


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


# ---------------------------------------------------------------------------
# Arc-length error computation
# ---------------------------------------------------------------------------

def _cid_map(case_ids_gt: np.ndarray) -> Dict[int, int]:
    return {int(cid): i for i, cid in enumerate(case_ids_gt)}


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
    """
    Returns {layout: (n_rows, M)} full SO(3) error in degrees.
    Requires R_est in shapes dict; raises RuntimeError if missing.
    """
    cmap = _cid_map(case_ids_gt)
    out: Dict[str, np.ndarray] = {}
    for lname, d in shapes.items():
        if "R_est" not in d:
            raise RuntimeError(
                f"R_est not found for layout '{lname}' in shapes NPZ.\n"
                "Re-run evaluate_kirchhoff_shape_estimation.py to regenerate it."
            )
        n, M, _, _ = d["R_est"].shape
        errs = np.full((n, M), np.nan)
        for k in range(n):
            gt_idx = cmap.get(int(d["case_ids"][k]))
            if gt_idx is None:
                continue
            R_est_k = d["R_est"][k]           # (M, 3, 3)
            R_gt_k  = orientations_gt[gt_idx] # (M, 3, 3)
            for i in range(M):
                errs[k, i] = np.degrees(
                    np.linalg.norm(_so3_log(R_est_k[i] @ R_gt_k[i].T))
                )
        out[lname] = errs
    return out


# ---------------------------------------------------------------------------
# Panel helpers
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
    """
    Grouped violin plot: two layout violins side-by-side for each metric.
    Body colour per layout, IQR bar + median dot overlaid.
    """
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
            ax.vlines(xpos, q25, q75,
                      color=COLORS[lname], linewidth=3.2, zorder=5)
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


def _panel_arclength(
    ax,
    errors: Dict[str, np.ndarray],
    s_grid: np.ndarray,
    layout_names: List[str],
    ylabel: str,
    scale_mm: bool = False,
) -> None:
    """Mean curve + 25–75 percentile shaded band."""
    for lname in layout_names:
        e    = errors[lname] * (1e3 if scale_mm else 1.0)
        mean = np.nanmean(e,           axis=0)
        p25  = np.nanpercentile(e, 25, axis=0)
        p75  = np.nanpercentile(e, 75, axis=0)
        col  = COLORS[lname]
        ax.plot(s_grid, mean, color=col, lw=1.8)
        ax.fill_between(s_grid, p25, p75, color=col, alpha=0.18, linewidth=0)

    # IMU location markers
    done: set = set()
    for lname in layout_names:
        for sp in _IMU_S.get(lname, []):
            if sp not in done:
                ax.axvline(sp, color="grey", lw=0.7, ls=":", alpha=0.55, zorder=0)
                done.add(sp)

    ax.set_xlabel("Normalised arc-length  $s$", fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_xlim(s_grid[0], s_grid[-1])
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, alpha=0.30, linestyle="--")
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Composite figure
# ---------------------------------------------------------------------------

def make_composite_figure(
    per_case: List[dict],
    layout_names: List[str],
    pos_errors: Dict[str, np.ndarray],
    ori_errors: Dict[str, np.ndarray],
    s_grid: np.ndarray,
    n_cases: int,
    n_noise: int,
    save_stem: Optional[Path] = None,
) -> None:
    """Build and save the 2×2 composite figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    # ── (a) position violin ────────────────────────────────────────────────
    _panel_violin(
        ax=ax_a,
        layout_names=layout_names,
        metric_keys=["mean_centerline_error", "tip_position_error"],
        per_case=per_case,
        scale_mm=True,
        ylabel="Error  [mm]",
        metric_xlabels=["Mean centerline\nerror", "Tip position\nerror"],
    )
    ax_a.set_title("Position metrics", fontsize=TITLE_FS)

    # ── (b) orientation violin ─────────────────────────────────────────────
    _panel_violin(
        ax=ax_b,
        layout_names=layout_names,
        metric_keys=["mean_orientation_error_deg", "tip_orientation_error_deg"],
        per_case=per_case,
        scale_mm=False,
        ylabel="Error  [deg]",
        metric_xlabels=["Mean orientation\nerror", "Tip orientation\nerror"],
    )
    ax_b.set_title("Orientation metrics", fontsize=TITLE_FS)

    # ── (c) arc-length position error ──────────────────────────────────────
    _panel_arclength(
        ax=ax_c,
        errors=pos_errors,
        s_grid=s_grid,
        layout_names=layout_names,
        ylabel="Position error  [mm]",
        scale_mm=True,
    )
    ax_c.set_title("Position error along arc-length", fontsize=TITLE_FS)

    # ── (d) arc-length orientation error ───────────────────────────────────
    _panel_arclength(
        ax=ax_d,
        errors=ori_errors,
        s_grid=s_grid,
        layout_names=layout_names,
        ylabel="Orientation error  [deg]",
        scale_mm=False,
    )
    ax_d.set_title("Orientation error along arc-length", fontsize=TITLE_FS)

    # ── panel labels (a)–(d) ──────────────────────────────────────────────
    for ax, lbl in zip([ax_a, ax_b, ax_c, ax_d],
                       ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(0.02, 0.98, lbl,
                transform=ax.transAxes,
                fontsize=PANEL_FS, fontweight="bold",
                va="top", ha="left")

    # ── shared legend ─────────────────────────────────────────────────────
    proxies = [mpatches.Patch(facecolor=COLORS[nm], alpha=0.75, label=nm)
               for nm in layout_names]
    fig.legend(
        handles=proxies,
        loc="lower center",
        ncol=len(layout_names),
        fontsize=LEGEND_FS,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.00),
    )

    # data-scope note inside one panel
    ax_d.text(
        0.98, 0.97,
        f"n = {n_cases} cases × {n_noise} realisations",
        transform=ax_d.transAxes,
        fontsize=8, ha="right", va="top", color="grey", style="italic",
    )

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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="2×2 composite manuscript figure for cross-model robustness "
                    "(no EKF re-run).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default="gt_data/results",
                        help="directory with kirchhoff_shape_est_results.json "
                             "and kirchhoff_shape_est_shapes.npz")
    parser.add_argument("--gt", default="gt_data/kirchhoff_gt_dataset.npz",
                        help="Step-1 ground-truth NPZ")
    parser.add_argument("--no-save", action="store_true",
                        help="show figure without writing files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    json_path   = results_dir / "kirchhoff_shape_est_results.json"
    shapes_path = results_dir / "kirchhoff_shape_est_shapes.npz"
    gt_path     = Path(args.gt)
    save_stem   = (None if args.no_save
                   else results_dir / "kirchhoff_shape_est_main_aggregate_violin")

    # ── load ──────────────────────────────────────────────────────────────
    for p in (json_path, shapes_path, gt_path):
        if not p.exists():
            raise FileNotFoundError(p)

    data_json    = _load_json(json_path)
    per_case     = data_json["per_case"]
    layout_names = list(data_json["summary"].keys())
    print(f"JSON: {len(per_case)} rows, layouts {layout_names}")

    positions_gt, orientations_gt, case_ids_gt = _load_gt(gt_path)
    n_cases = len(positions_gt)
    M = positions_gt.shape[1]
    print(f"GT: {n_cases} cases, {M} arc-length pts")

    _, shapes = _load_shapes(shapes_path)
    n_noise = len(shapes[layout_names[0]]["p_est"]) // n_cases
    has_R   = "R_est" in shapes[layout_names[0]]
    s_grid  = np.linspace(0.0, 1.0, M)
    print(f"Shapes NPZ: n_noise≈{n_noise}, R_est={has_R}")

    # ── compute arc-length errors ──────────────────────────────────────────
    print("Computing arc-length position errors …")
    pos_errors = compute_arclength_position_errors(shapes, positions_gt, case_ids_gt)

    print("Computing arc-length SO(3) orientation errors …")
    ori_errors = compute_arclength_orientation_errors(
        shapes, orientations_gt, case_ids_gt
    )

    # ── summary ───────────────────────────────────────────────────────────
    print(f"\nData scope: {n_cases} cases × {n_noise} realisations "
          f"= {n_cases * n_noise} rows per layout")
    for lname in layout_names:
        me = np.nanmean(pos_errors[lname]) * 1e3
        oe = np.nanmean(ori_errors[lname])
        print(f"  {lname:<8s}  mean pos err = {me:.3f} mm   "
              f"mean ori err = {oe:.3f} deg")

    # ── build figure ───────────────────────────────────────────────────────
    print("\nBuilding 2×2 composite figure …")
    make_composite_figure(
        per_case     = per_case,
        layout_names = layout_names,
        pos_errors   = pos_errors,
        ori_errors   = ori_errors,
        s_grid       = s_grid,
        n_cases      = n_cases,
        n_noise      = n_noise,
        save_stem    = save_stem,
    )


if __name__ == "__main__":
    main()
