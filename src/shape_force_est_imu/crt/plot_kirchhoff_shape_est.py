#!/usr/bin/env python3
"""
Plot-only companion to evaluate_kirchhoff_shape_estimation.py.

Reads saved results (JSON + shapes NPZ + GT NPZ) and regenerates figures
without re-running the EKF.

Three figure types
------------------
  --plot-summary   — aggregate bar chart: mean centerline error & tip error
                     (reads kirchhoff_shape_est_results.json)
  --plot-overlays  — representative shape overlays GT vs EKF per layout
                     (reads kirchhoff_shape_est_shapes.npz + GT NPZ)
  --plot-metrics   — per-layout violin plots for all five geometry metrics
                     (reads kirchhoff_shape_est_results.json)

Usage
-----
  python plot_kirchhoff_shape_est.py
  python plot_kirchhoff_shape_est.py --results-dir gt_data/results --plot-summary
  python plot_kirchhoff_shape_est.py --plot-summary --plot-overlays --plot-metrics
  python plot_kirchhoff_shape_est.py --num-overlay-cases 5 --no-save
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_gt_positions(gt_npz: Path) -> np.ndarray:
    """Returns positions (N, M, 3) and case_ids (N,)."""
    data = np.load(gt_npz, allow_pickle=True)
    return data["positions"], data["case_id"]


def load_shapes(shapes_npz: Path) -> dict:
    """
    Returns dict keyed by layout_name with entries:
      p_est     : (n_rows, M, 3)
      case_ids  : (n_rows,)
      noise_real: (n_rows,)
    """
    data = np.load(shapes_npz, allow_pickle=True)
    layout_names = list(data["layout_names"])
    shapes: dict = {}
    for lname in layout_names:
        shapes[lname] = {
            "p_est"     : data[f"p_est_{lname}"],
            "case_ids"  : data[f"case_ids_{lname}"],
            "noise_real": data[f"noise_real_{lname}"],
        }
    return shapes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tight_axis_limits(ax, positions_list: list[np.ndarray],
                       pad_frac: float = 0.07) -> None:
    all_pts = np.vstack(positions_list)
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
            fig.savefig(p, dpi=150)
            print(f"Figure saved → {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — aggregate bar chart
# ---------------------------------------------------------------------------

def plot_cross_model_summary(
    summary: Dict[str, Dict],
    save_stem: Optional[Path] = None,
) -> None:
    """Two-panel bar plot: mean centerline error and tip position error."""
    names  = list(summary.keys())
    n      = len(names)
    x      = np.arange(n)
    colors = plt.cm.tab10(np.linspace(0, 0.5, n))

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
    _save_show(fig, save_stem, "summary")


# ---------------------------------------------------------------------------
# Figure 2 — shape overlays
# ---------------------------------------------------------------------------

def plot_representative_overlays(
    positions_gt: np.ndarray,
    case_ids_gt: np.ndarray,
    shapes: Dict[str, dict],
    n_cases: int = 3,
    save_stem: Optional[Path] = None,
) -> None:
    """
    Shape overlay for n_cases representative cases (low / median / high error).

    Representative cases are selected by mean centerline error of the first
    layout, noise-realization 0.
    """
    layout_names = list(shapes.keys())
    if not layout_names:
        print("No shape data available for overlays.")
        return

    cid_to_idx = {int(cid): idx for idx, cid in enumerate(case_ids_gt)}

    # Rank cases by tip-to-tip distance (proxy: use GT tip z-spread as a
    # rough error ordering) — better: use per-case tip error from the JSON.
    # Here we rank by mean p_est deviation for the first layout, noise_real=0.
    lname0   = layout_names[0]
    d0       = shapes[lname0]
    mask0    = d0["noise_real"] == 0
    c0       = d0["case_ids"][mask0]
    p0       = d0["p_est"][mask0]      # (N, M, 3)

    case_err: Dict[int, float] = {}
    for k, cid in enumerate(c0):
        gt_idx = cid_to_idx.get(int(cid))
        if gt_idx is None:
            continue
        p_gt = positions_gt[gt_idx]    # (M, 3)
        err  = float(np.mean(np.linalg.norm(p0[k] - p_gt, axis=1)))
        case_err[int(cid)] = err

    sorted_cases = sorted(case_err.items(), key=lambda x: x[1])
    n_total  = len(sorted_cases)
    pick_idx = [0, n_total // 2, n_total - 1][:n_cases]
    rep_cases = [sorted_cases[i][0] for i in pick_idx]
    labels    = ["low error", "median error", "high error"][:n_cases]

    layout_colors = plt.cm.Set1(np.linspace(0, 0.55, len(layout_names)))

    fig = plt.figure(figsize=(5 * n_cases, 5))
    for col, (cid, lbl) in enumerate(zip(rep_cases, labels)):
        ax     = fig.add_subplot(1, n_cases, col + 1, projection="3d")
        gt_idx = cid_to_idx[cid]
        p_gt   = positions_gt[gt_idx]

        ax.plot(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2],
                "k-", lw=1.8, label="GT (Kirchhoff)")

        all_pos = [p_gt]
        for li, lname in enumerate(layout_names):
            d      = shapes[lname]
            mask   = (d["case_ids"] == cid) & (d["noise_real"] == 0)
            if not np.any(mask):
                continue
            p_est  = d["p_est"][mask][0]
            ax.plot(p_est[:, 0], p_est[:, 1], p_est[:, 2],
                    "--", color=layout_colors[li], lw=1.3,
                    label=lname, alpha=0.9)
            all_pos.append(p_est)

        ax.set_title(f"case {cid}  ({lbl})", fontsize=8)
        ax.set_xlabel("x [m]", fontsize=7)
        ax.set_ylabel("y [m]", fontsize=7)
        ax.set_zlabel("z [m]", fontsize=7)
        ax.set_box_aspect([1, 1, 1])
        _tight_axis_limits(ax, all_pos)
        if col == 0:
            ax.legend(fontsize=6, loc="upper left")

    fig.suptitle("Shape overlays — GT vs EKF estimate (cross-model)", fontsize=10)
    _save_show(fig, save_stem, "overlays")


# ---------------------------------------------------------------------------
# Figure 3 — per-metric violin plots
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "mean_centerline_error"     : "Mean centerline error [mm]",
    "max_centerline_error"      : "Max centerline error [mm]",
    "tip_position_error"        : "Tip position error [mm]",
    "tip_orientation_error_deg" : "Tip orientation error [deg]",
    "mean_orientation_error_deg": "Mean orientation error [deg]",
}
MM_METRICS = {
    "mean_centerline_error",
    "max_centerline_error",
    "tip_position_error",
}


def plot_metric_distributions(
    per_case: List[Dict],
    save_stem: Optional[Path] = None,
) -> None:
    """Violin plot of each geometry metric, grouped by layout."""
    from collections import defaultdict

    groups: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in per_case:
        lname = row["layout_name"]
        for mk in METRIC_LABELS:
            val = row[mk]
            if mk in MM_METRICS:
                val *= 1e3   # → mm
            groups[lname][mk].append(val)

    layout_names = list(groups.keys())
    n_metrics    = len(METRIC_LABELS)
    metric_keys  = list(METRIC_LABELS.keys())

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(layout_names)))

    for ax, mk in zip(axes, metric_keys):
        data_per_layout = [groups[ln][mk] for ln in layout_names]
        parts = ax.violinplot(data_per_layout, showmedians=True, showextrema=True)
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.70)
        for comp in ("cmedians", "cmins", "cmaxes", "cbars"):
            if comp in parts:
                parts[comp].set_color("k")
                parts[comp].set_linewidth(0.9)

        ax.set_xticks(range(1, len(layout_names) + 1))
        ax.set_xticklabels(layout_names, rotation=12, ha="right", fontsize=8)
        ax.set_ylabel(METRIC_LABELS[mk], fontsize=8)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle("Metric distributions — cross-model EKF (Kirchhoff rod)", fontsize=10)
    _save_show(fig, save_stem, "metrics")


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
    parser.add_argument(
        "--json", default=None,
        help="explicit path to kirchhoff_shape_est_results.json "
             "(overrides --results-dir for summary/metrics)",
    )
    parser.add_argument(
        "--shapes", default=None,
        help="explicit path to kirchhoff_shape_est_shapes.npz "
             "(overrides --results-dir for overlays)",
    )
    parser.add_argument(
        "--gt", default="gt_data/kirchhoff_gt_dataset.npz",
        help="Step-1 ground-truth NPZ (required for --plot-overlays)",
    )
    parser.add_argument("--plot-summary",  action="store_true", default=True)
    parser.add_argument("--no-plot-summary", dest="plot_summary",
                        action="store_false")
    parser.add_argument("--plot-overlays", action="store_true", default=False)
    parser.add_argument("--plot-metrics",  action="store_true", default=False)
    parser.add_argument("--num-overlay-cases", type=int, default=3)
    parser.add_argument("--no-save", action="store_true",
                        help="display figures but do not save PNG/PDF")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    json_path   = Path(args.json)   if args.json   else results_dir / "kirchhoff_shape_est_results.json"
    shapes_path = Path(args.shapes) if args.shapes else results_dir / "kirchhoff_shape_est_shapes.npz"
    gt_path     = Path(args.gt)

    # Resolve save stem
    save_stem = None if args.no_save else (results_dir / "kirchhoff_shape_est")

    # ------------------------------------------------------------------ #
    # Load JSON (summary + per-case metrics)
    # ------------------------------------------------------------------ #
    if not json_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {json_path}")
    data_json = load_json(json_path)
    summary   = data_json["summary"]
    per_case  = data_json["per_case"]
    print(f"Loaded {len(per_case)} per-case rows from {json_path}")
    print(f"Layouts: {list(summary.keys())}")

    # ------------------------------------------------------------------ #
    # Summary bar chart
    # ------------------------------------------------------------------ #
    if args.plot_summary:
        print("\n--- summary bar chart ---")
        plot_cross_model_summary(summary, save_stem=save_stem)

    # ------------------------------------------------------------------ #
    # Metric distributions
    # ------------------------------------------------------------------ #
    if args.plot_metrics:
        print("\n--- metric distributions ---")
        plot_metric_distributions(per_case, save_stem=save_stem)

    # ------------------------------------------------------------------ #
    # Shape overlays (needs shapes NPZ + GT NPZ)
    # ------------------------------------------------------------------ #
    if args.plot_overlays:
        if not shapes_path.exists():
            print(f"WARNING: shapes NPZ not found at {shapes_path} — "
                  "skipping overlays.\n"
                  "Re-run evaluate_kirchhoff_shape_estimation.py to generate it.")
        elif not gt_path.exists():
            print(f"WARNING: GT NPZ not found at {gt_path} — "
                  "skipping overlays.")
        else:
            print("\n--- shape overlays ---")
            positions_gt, case_ids_gt = load_gt_positions(gt_path)
            shapes = load_shapes(shapes_path)
            print(f"Loaded {len(positions_gt)} GT cases from {gt_path}")
            for lname, d in shapes.items():
                print(f"  {lname}: {len(d['p_est'])} estimated shapes")
            plot_representative_overlays(
                positions_gt  = positions_gt,
                case_ids_gt   = case_ids_gt,
                shapes        = shapes,
                n_cases       = args.num_overlay_cases,
                save_stem     = save_stem,
            )


if __name__ == "__main__":
    main()
