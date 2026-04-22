#!/usr/bin/env python3
"""
Plot-only companion for kirchhoff wrench estimation results (Step 4).

Reads the CSV / JSON produced by evaluate_kirchhoff_wrench_estimation.py
and generates:

  Figure 1 — Violin comparison  (2×2 grid: {Force, Moment} × {2-IMU, 3-IMU})
             x-axis: method (Direct | MAP); y-axis: absolute error
  Figure 2 — Error vs wrench magnitude scatter (force)
  Figure 3 — Direction error violin  (force direction, moment direction)
  Figure 4 — Horizontal bar chart: mean ± std for all four (layout, method) combos

Usage
-----
  cd src/shape_force_est_imu/crt
  python plot_kirchhoff_wrench_estimation_summary.py
  python plot_kirchhoff_wrench_estimation_summary.py \\
      --results gt_data/results/kirchhoff_wrench_est_results.csv \\
      --save-dir gt_data/results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
LABEL_FS  = 13
TICK_FS   = 11
TITLE_FS  = 13
PANEL_FS  = 14
LEGEND_FS = 11
ANN_FS    = 9.0
LW        = 1.5

_METHOD_COLORS = {
    "direct": "#1f77b4",
    "map":    "#d62728",
}
_METHOD_LABELS = {
    "direct": "Direct",
    "map":    "MAP",
}
_LAYOUT_ORDER  = ["2-IMU", "3-IMU"]
_METHOD_ORDER  = ["direct", "map"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv_results(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def cast_floats(rows: list[dict], float_keys: list[str]) -> list[dict]:
    for r in rows:
        for k in float_keys:
            if k in r and r[k] not in ("", None):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    r[k] = float("nan")
    return rows


def extract_column(rows: list[dict], layout: str, method: str, key: str) -> np.ndarray:
    vals = [r[key] for r in rows
            if r["layout_name"] == layout and r["method"] == method]
    return np.array([float(v) for v in vals if v not in ("", None, "nan")])


# ---------------------------------------------------------------------------
# Violin helper
# ---------------------------------------------------------------------------

def _violin(ax, positions, datasets, colors, labels, width=0.38):
    for pos, data, color, label in zip(positions, datasets, colors, labels):
        if len(data) == 0:
            continue
        vp = ax.violinplot(data, positions=[pos], widths=width,
                           showmedians=True, showextrema=False)
        for pc in vp["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
            pc.set_edgecolor(color)
        vp["cmedians"].set_color(color)
        vp["cmedians"].set_linewidth(2.2)
        ax.scatter([pos], [np.mean(data)], marker="D", s=28,
                   color=color, zorder=5, label=label)


# ---------------------------------------------------------------------------
# Figure 1: violin force + moment per layout
# ---------------------------------------------------------------------------

def fig_violin_comparison(rows: list[dict], save_stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey="row")
    fig.suptitle("Tip-Wrench Estimation Errors: Direct vs MAP",
                 fontsize=TITLE_FS + 1, fontweight="bold")

    row_labels  = ["Force error [mN]", "Moment error [mN·m]"]
    err_keys    = ["force_err_N", "moment_err_Nm"]
    err_scales  = [1e3, 1e3]

    for col_idx, layout in enumerate(_LAYOUT_ORDER):
        for row_idx, (err_key, ey_label, scale) in \
                enumerate(zip(err_keys, row_labels, err_scales)):

            ax = axes[row_idx][col_idx]
            positions   = [1, 2]
            datasets    = [extract_column(rows, layout, m, err_key) * scale
                           for m in _METHOD_ORDER]
            colors  = [_METHOD_COLORS[m] for m in _METHOD_ORDER]
            labels  = [_METHOD_LABELS[m] for m in _METHOD_ORDER]

            _violin(ax, positions, datasets, colors, labels, width=0.55)

            if row_idx == 0:
                ax.set_title(layout, fontsize=TITLE_FS, fontweight="bold")
            ax.set_xticks([1, 2])
            ax.set_xticklabels(labels, fontsize=TICK_FS)
            ax.set_ylabel(ey_label, fontsize=LABEL_FS)
            ax.tick_params(axis="y", labelsize=TICK_FS)
            ax.set_xlim(0.4, 2.6)
            ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
            ax.set_axisbelow(True)

            # Annotate mean
            for pos, data, color in zip(positions, datasets, colors):
                if len(data):
                    ax.text(pos, np.mean(data) * 1.04,
                            f"{np.mean(data):.1f}",
                            ha="center", va="bottom",
                            fontsize=ANN_FS, color=color, fontweight="bold")

    # Legend (top-right subplot only)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="D", color=_METHOD_COLORS[m],
                      label=_METHOD_LABELS[m], linewidth=0,
                      markersize=7) for m in _METHOD_ORDER]
    axes[0][1].legend(handles=handles, fontsize=LEGEND_FS, loc="upper right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = save_stem.parent / (save_stem.name + "_violin." + ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: direction-error violin
# ---------------------------------------------------------------------------

def fig_direction_errors(rows: list[dict], save_stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle("Tip-Wrench Direction Errors: Direct vs MAP",
                 fontsize=TITLE_FS + 1, fontweight="bold")

    dir_keys   = ["force_dir_err_deg", "moment_dir_err_deg"]
    dir_labels = ["Force direction error [°]", "Moment direction error [°]"]

    for ax, key, ylabel in zip(axes, dir_keys, dir_labels):
        pos  = 1
        all_groups = []
        all_colors = []
        all_labels = []
        all_pos    = []
        tick_labels = []
        tick_pos    = []
        offset = 0

        for layout in _LAYOUT_ORDER:
            group_center = offset + 1.5
            tick_pos.append(group_center)
            tick_labels.append(layout)
            for mi, method in enumerate(_METHOD_ORDER):
                p = offset + mi + 1
                data = extract_column(rows, layout, method, key)
                all_groups.append(data)
                all_colors.append(_METHOD_COLORS[method])
                all_labels.append(f"{layout} {_METHOD_LABELS[method]}")
                all_pos.append(p)
            offset += 3

        _violin(ax, all_pos, all_groups, all_colors, all_labels, width=0.55)

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=TICK_FS)
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=_METHOD_COLORS[m], label=_METHOD_LABELS[m],
                     alpha=0.7) for m in _METHOD_ORDER]
    axes[1].legend(handles=handles, fontsize=LEGEND_FS)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = save_stem.parent / (save_stem.name + "_direction." + ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: bar chart summary
# ---------------------------------------------------------------------------

def fig_bar_summary(rows: list[dict], save_stem: Path) -> None:
    combos   = [(lay, mth) for lay in _LAYOUT_ORDER for mth in _METHOD_ORDER]
    n        = len(combos)
    x        = np.arange(n)
    bar_w    = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Mean Wrench Estimation Error (±1 std)",
                 fontsize=TITLE_FS + 1, fontweight="bold")

    for ax, key, ylabel, scale, unit in [
        (axes[0], "force_err_N",   "Force error",   1e3, "mN"),
        (axes[1], "moment_err_Nm", "Moment error",  1e3, "mN·m"),
    ]:
        means, stds, colors = [], [], []
        for layout, method in combos:
            vals = extract_column(rows, layout, method, key) * scale
            means.append(np.mean(vals) if len(vals) else 0.0)
            stds.append(np.std(vals)   if len(vals) else 0.0)
            colors.append(_METHOD_COLORS[method])

        bars = ax.bar(x, means, bar_w, yerr=stds, capsize=4,
                      color=colors, alpha=0.80, edgecolor="k", linewidth=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{lay}\n{_METHOD_LABELS[mth]}" for lay, mth in combos],
            fontsize=TICK_FS)
        ax.set_ylabel(f"{ylabel} [{unit}]", fontsize=LABEL_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean * 1.05,
                    f"{mean:.1f}", ha="center", va="bottom",
                    fontsize=ANN_FS, fontweight="bold")

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=_METHOD_COLORS[m], label=_METHOD_LABELS[m],
                     alpha=0.8) for m in _METHOD_ORDER]
    axes[1].legend(handles=handles, fontsize=LEGEND_FS, loc="upper right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = save_stem.parent / (save_stem.name + "_bars." + ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: error vs GT wrench magnitude scatter
# ---------------------------------------------------------------------------

def fig_error_vs_magnitude(rows: list[dict], save_stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.suptitle("Wrench Error vs GT Magnitude",
                 fontsize=TITLE_FS + 1, fontweight="bold")

    for col_idx, layout in enumerate(_LAYOUT_ORDER):
        for method, marker, ms in [("direct", "o", 18), ("map", "^", 20)]:
            for row_idx, (mag_key, err_key, xlabel, ylabel, scale) in enumerate([
                ("force_gt_norm_N",   "force_err_N",
                 "GT force magnitude [N]",  "Force error [mN]", 1e3),
                ("moment_gt_norm_Nm", "moment_err_Nm",
                 "GT moment magnitude [N·m]", "Moment error [mN·m]", 1e3),
            ]):
                ax   = axes[row_idx][col_idx]
                mags = extract_column(rows, layout, method, mag_key)
                errs = extract_column(rows, layout, method, err_key) * scale
                ax.scatter(mags, errs,
                           s=ms, marker=marker,
                           color=_METHOD_COLORS[method],
                           alpha=0.35, label=_METHOD_LABELS[method])

        for row_idx, (xlabel, ylabel) in enumerate([
            ("GT force magnitude [N]",    "Force error [mN]"),
            ("GT moment magnitude [N·m]", "Moment error [mN·m]"),
        ]):
            ax = axes[row_idx][col_idx]
            ax.set_xlabel(xlabel, fontsize=LABEL_FS - 1)
            ax.set_ylabel(ylabel, fontsize=LABEL_FS - 1)
            ax.tick_params(labelsize=TICK_FS)
            if row_idx == 0:
                ax.set_title(layout, fontsize=TITLE_FS, fontweight="bold")
            if col_idx == 0 and row_idx == 0:
                ax.legend(fontsize=LEGEND_FS)
            ax.yaxis.grid(True, linewidth=0.5, alpha=0.6)
            ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = save_stem.parent / (save_stem.name + "_scatter." + ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot wrench estimation results from Step 4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        default="gt_data/results/kirchhoff_wrench_est_results.csv",
        help="CSV output from evaluate_kirchhoff_wrench_estimation.py",
    )
    parser.add_argument(
        "--save-dir",
        default="gt_data/results",
        help="directory to write figures",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    def _resolve(p: str, must_exist: bool = True) -> Path:
        raw = Path(p)
        out = raw if raw.is_absolute() else (script_dir / raw).resolve()
        if must_exist and not out.exists():
            raise FileNotFoundError(out)
        return out

    csv_path = _resolve(args.results)
    save_dir = _resolve(args.save_dir, must_exist=False)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_stem = save_dir / "kirchhoff_wrench_est"

    float_keys = [
        "force_err_N", "moment_err_Nm",
        "force_dir_err_deg", "moment_dir_err_deg",
        "nrmse_force", "nrmse_moment",
        "force_gt_norm_N", "moment_gt_norm_Nm",
    ]

    print(f"Loading results from {csv_path} ...")
    rows = load_csv_results(csv_path)
    rows = cast_floats(rows, float_keys)
    print(f"  {len(rows)} rows  "
          f"({len([r for r in rows if r['method']=='direct'])} direct, "
          f"{len([r for r in rows if r['method']=='map'])} MAP)")

    # Quick console summary
    for layout in _LAYOUT_ORDER:
        for method in _METHOD_ORDER:
            ferr = extract_column(rows, layout, method, "force_err_N") * 1e3
            merr = extract_column(rows, layout, method, "moment_err_Nm") * 1e3
            if len(ferr):
                print(f"  {layout:<8} {_METHOD_LABELS[method]:<7}  "
                      f"force {np.mean(ferr):6.2f} mN   "
                      f"moment {np.mean(merr):7.3f} mN·m")

    print("\nGenerating figures ...")
    fig_violin_comparison(rows, save_stem)
    fig_direction_errors(rows, save_stem)
    fig_bar_summary(rows, save_stem)
    fig_error_vs_magnitude(rows, save_stem)
    print("Done.")


if __name__ == "__main__":
    main()
