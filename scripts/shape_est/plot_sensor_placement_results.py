#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_sensor_placement_results.py
=================================
Plotting-only script for the finalized matched-model sensor placement study.
Reads already-saved CSV / JSON results and generates manuscript-ready figures
WITHOUT rerunning any simulations.

Input files (produced by sensor_placement_study_final.py)
---------------------------------------------------------
  sensor_placement_results.csv   -- all 17 configurations
  sensor_best_by_count.csv       -- best config per sensor count
  sensor_placement_results.json  -- full metadata + results

Output figures
--------------
  A. best_vs_count.png / .pdf      -- best RMSE vs number of IMUs
  B. heatmap_2imu.png  / .pdf      -- sparse 2-IMU placement heatmap
                                      (built from the 6 predefined 2-IMU configs)
  C. bar_all_configs.png / .pdf    -- horizontal bar chart (optional)

Heatmap note
------------
The full 8x8 or 9x9 grid sweep is NOT saved in the CSV (it was an optional,
separate run in the simulation script).  This script reconstructs a sparse
4x4 heatmap from the 6 predefined 2-IMU configs:
  positions = {0.25, 0.50, 0.75, 1.00}
  valid pairs: all (s1, s2) with s1 < s2

Axis-centering fix: uses pcolormesh with cell-EDGE coordinates constructed
from the sensor-position grid, then sets xticks / yticks to the center values.
This guarantees that e.g. the label "0.25" appears at the CENTER of its cell,
not on a boundary.

Usage
-----
  python plot_sensor_placement_results.py --plot-all
  python plot_sensor_placement_results.py --plot-best-vs-count
  python plot_sensor_placement_results.py --plot-heatmap
  python plot_sensor_placement_results.py --plot-all --output-dir figures
"""

import argparse
import ast
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
#  STYLE CONSTANTS
# ============================================================

FS_LABEL  = 16
FS_TICK   = 13
FS_LEGEND = 12
FS_TITLE  = 15
LW        = 2.0
MS        = 7

_COUNT_COLOR = {1: "#4878d0", 2: "#6acc65", 3: "#d65f5f",
                4: "#b47cc7", 5: "#c4ad66"}
_COUNT_LABEL = {1: "1 IMU", 2: "2 IMUs", 3: "3 IMUs",
                4: "4 IMUs", 5: "5 IMUs"}


# ============================================================
#  DATA LOADERS
# ============================================================

def load_all_configs(csv_path: str) -> List[Dict]:
    """Load sensor_placement_results.csv into a list of dicts."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["n_sensors"]           = int(row["n_sensors"])
            row["rmse_final_mean"]     = float(row["rmse_final_mean"])
            row["rmse_final_std"]      = float(row["rmse_final_std"])
            row["rmse_mean_mean"]      = float(row["rmse_mean_mean"])
            row["nis_mean_postburnin"] = float(row["nis_mean_postburnin"])
            row["nis_dof"]             = int(row["nis_dof"])
            row["imu_positions"]       = ast.literal_eval(row["imu_positions"])
            rows.append(row)
    return rows


def load_best_by_count(csv_path: str) -> List[Dict]:
    """Load sensor_best_by_count.csv into a list of dicts."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["n_sensors"]           = int(row["n_sensors"])
            row["rmse_final_mean"]     = float(row["rmse_final_mean"])
            row["rmse_final_std"]      = float(row["rmse_final_std"])
            row["nis_mean_postburnin"] = float(row["nis_mean_postburnin"])
            row["nis_dof"]             = int(row["nis_dof"])
            row["imu_positions"]       = ast.literal_eval(row["imu_positions"])
            rows.append(row)
    return rows


def load_json_meta(json_path: str) -> Dict:
    with open(json_path) as f:
        return json.load(f)


# ============================================================
#  FIGURE A: BEST RMSE vs SENSOR COUNT
# ============================================================

def plot_best_vs_count(best_rows: List[Dict],
                       output_dir: str,
                       fmt: str = "png") -> None:
    """
    Plot best final RMSE vs number of IMUs with error bars.
    One colored point per sensor count.
    """
    best_rows = sorted(best_rows, key=lambda r: r["n_sensors"])
    counts = [r["n_sensors"]       for r in best_rows]
    means  = [r["rmse_final_mean"] for r in best_rows]
    stds   = [r["rmse_final_std"]  for r in best_rows]
    colors = [_COUNT_COLOR.get(n, "gray") for n in counts]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Error bar spine
    ax.errorbar(counts, means, yerr=stds,
                fmt="none", ecolor="#555555",
                capsize=5, capthick=1.5, elinewidth=1.2, zorder=2)

    # Connect points
    ax.plot(counts, means, "-", color="#444444",
            linewidth=LW, zorder=2, alpha=0.5)

    # Colored dots per count
    for n, m, c in zip(counts, means, colors):
        ax.scatter([n], [m], color=c, s=MS ** 2, zorder=4,
                   edgecolors="white", linewidths=0.8)

    # Annotations: config label
    for row in best_rows:
        n   = row["n_sensors"]
        m   = row["rmse_final_mean"]
        pos = row["imu_positions"]
        lbl = "[" + ", ".join(f"{v:.2f}" for v in pos) + "]"
        ax.annotate(lbl, xy=(n, m),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color="#333333")

    ax.set_xticks(counts)
    ax.set_xticklabels([str(n) for n in counts], fontsize=FS_TICK)
    ax.set_xlabel("Number of IMUs", fontsize=FS_LABEL)
    ax.set_ylabel("Best final RMSE (modal coefficients)", fontsize=FS_LABEL)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(True, alpha=0.35, which="both")
    fig.tight_layout()

    _save(fig, output_dir, "best_vs_count", fmt)
    plt.show()


# ============================================================
#  FIGURE B: SPARSE 2-IMU HEATMAP
# ============================================================

def plot_heatmap_2imu(all_rows: List[Dict],
                      output_dir: str,
                      fmt: str = "png") -> None:
    """
    Build a sparse heatmap from the 6 predefined 2-IMU configurations.

    The 4 sensor positions {0.25, 0.50, 0.75, 1.00} define a 4x4 grid.
    Only cells where s1 < s2 have data (upper triangle); symmetric fill
    is applied so the map looks symmetric.  Diagonal cells are masked.

    Axis-centering fix
    ------------------
    pcolormesh requires EDGE coordinates, not center coordinates.
    Given center positions p = [0.25, 0.50, 0.75, 1.00], the edges are
    constructed by taking midpoints between adjacent centers and adding
    half-steps on each end:
        edges[0]   = p[0] - (p[1] - p[0]) / 2
        edges[k]   = (p[k-1] + p[k]) / 2   for k = 1 .. N-1
        edges[N]   = p[-1] + (p[-1] - p[-2]) / 2
    Then xticks / yticks are set to p (the center values), ensuring each
    label appears at the center of its corresponding cell.
    """
    two_imu = [r for r in all_rows if r["n_sensors"] == 2]
    if not two_imu:
        print("  [heatmap skipped] No 2-IMU data found in CSV.")
        return

    # Collect all unique positions that appear across 2-IMU configs
    pos_set = set()
    for r in two_imu:
        for p in r["imu_positions"]:
            pos_set.add(round(p, 4))
    positions = sorted(pos_set)
    N = len(positions)
    pos_idx = {p: i for i, p in enumerate(positions)}

    # Build N x N RMSE grid (NaN = no data / invalid)
    grid = np.full((N, N), np.nan)

    for r in two_imu:
        p1, p2 = [round(v, 4) for v in r["imu_positions"]]
        i, j   = pos_idx[p1], pos_idx[p2]
        val    = r["rmse_final_mean"]
        grid[i, j] = val
        grid[j, i] = val   # symmetric fill

    # Mask the diagonal (s1 == s2, physically invalid)
    diag_mask = np.zeros((N, N), dtype=bool)
    np.fill_diagonal(diag_mask, True)

    # --- Build cell-edge coordinates from center positions ---
    # This is the fix: pcolormesh plots cells whose boundaries are at `edges`,
    # so ticks placed at `positions` will land at the center of each cell.
    pos_arr = np.array(positions)
    edges   = np.empty(N + 1)
    edges[0]    = pos_arr[0] - (pos_arr[1] - pos_arr[0]) / 2.0
    edges[1:-1] = (pos_arr[:-1] + pos_arr[1:]) / 2.0
    edges[-1]   = pos_arr[-1] + (pos_arr[-1] - pos_arr[-2]) / 2.0

    # Colormap: mask NaN as light grey, diagonal as dark grey
    cmap = plt.cm.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color="#dddddd")   # NaN → light grey

    # Compute vmin / vmax from valid (non-diagonal, non-NaN) cells
    valid_vals = grid[~diag_mask & ~np.isnan(grid)]
    vmin, vmax = valid_vals.min(), valid_vals.max()

    # Mask diagonal cells so they render in the bad-data colour
    grid_masked = np.ma.array(grid, mask=diag_mask | np.isnan(grid))

    fig, ax = plt.subplots(figsize=(6, 5))

    # pcolormesh with edge coordinates — tick labels will land on cell centers
    pcm = ax.pcolormesh(edges, edges, grid_masked,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        shading="flat")

    # Diagonal hatching to mark invalid cells clearly
    for k in range(N):
        x_lo, x_hi = edges[k], edges[k + 1]
        y_lo, y_hi = edges[k], edges[k + 1]
        rect = mpatches.Rectangle(
            (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
            linewidth=0, facecolor="#aaaaaa", hatch="//", alpha=0.6, zorder=3
        )
        ax.add_patch(rect)

    # Annotate each valid cell with its RMSE value
    for i in range(N):
        for j in range(N):
            if i == j or np.isnan(grid[i, j]):
                continue
            cx = (edges[j] + edges[j + 1]) / 2.0   # note: j indexes X-axis
            cy = (edges[i] + edges[i + 1]) / 2.0   # i indexes Y-axis
            ax.text(cx, cy, f"{grid[i, j]:.4f}",
                    ha="center", va="center",
                    fontsize=8, color="black", zorder=5)

    # Colorbar
    cb = plt.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("Mean final RMSE", fontsize=FS_LEGEND)
    cb.ax.tick_params(labelsize=FS_TICK - 2)

    # Axes — ticks at sensor position CENTERS
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels([f"{p:.2f}" for p in positions], fontsize=FS_TICK)
    ax.set_yticklabels([f"{p:.2f}" for p in positions], fontsize=FS_TICK)
    ax.set_xlabel("Sensor 2 position $s_2$", fontsize=FS_LABEL)
    ax.set_ylabel("Sensor 1 position $s_1$", fontsize=FS_LABEL)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(edges[0], edges[-1])
    ax.set_aspect("equal")

    n_data = len(two_imu)
    ax.set_title(f"2-IMU placement  ({n_data} configurations; lower = better)",
                 fontsize=FS_TITLE - 1)

    fig.tight_layout()
    _save(fig, output_dir, "heatmap_2imu", fmt)
    plt.show()

    # Inform the user what's missing for a full grid sweep
    print(f"\n  [heatmap note] This heatmap shows the {n_data} predefined 2-IMU "
          "configurations from the study.")
    print("  For a full grid sweep (e.g. 8x8), re-run:")
    print("    python sensor_placement_study_final.py --heatmap-grid 8 --no-heatmap False")


# ============================================================
#  FIGURE C: HORIZONTAL BAR CHART (all configs)
# ============================================================

def plot_bar_all_configs(all_rows: List[Dict],
                         output_dir: str,
                         fmt: str = "png") -> None:
    """Horizontal bar chart of final RMSE for all configurations."""
    names  = [r["config"]          for r in all_rows]
    means  = [r["rmse_final_mean"] for r in all_rows]
    stds   = [r["rmse_final_std"]  for r in all_rows]
    counts = [r["n_sensors"]       for r in all_rows]
    colors = [_COUNT_COLOR.get(c, "gray") for c in counts]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.42)))

    ax.barh(y, means, xerr=stds, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=FS_TICK - 2)
    ax.set_xlabel("Final RMSE mean (+/- std)", fontsize=FS_LABEL)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    legend_handles = [mpatches.Patch(color=_COUNT_COLOR[n],
                                     label=_COUNT_LABEL[n])
                      for n in sorted(_COUNT_COLOR) if n in set(counts)]
    ax.legend(handles=legend_handles, fontsize=FS_LEGEND,
              loc="lower right", framealpha=0.9)

    fig.tight_layout()
    _save(fig, output_dir, "bar_all_configs", fmt)
    plt.show()


# ============================================================
#  SAVE HELPER
# ============================================================

def _save(fig: plt.Figure, output_dir: str,
          stem: str, fmt: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{stem}.{fmt}")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    print(f"  Saved -> {path}")
    # Always save PDF alongside
    if fmt != "pdf":
        pdf_path = os.path.join(output_dir, f"{stem}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"  Saved -> {pdf_path}")


# ============================================================
#  MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot sensor placement results from saved CSV/JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-csv",  default="sensor_placement_results.csv",
                        help="Path to sensor_placement_results.csv")
    parser.add_argument("--best-csv",   default="sensor_best_by_count.csv",
                        help="Path to sensor_best_by_count.csv")
    parser.add_argument("--input-json", default="sensor_placement_results.json",
                        help="Path to sensor_placement_results.json (for metadata)")
    parser.add_argument("--output-dir", default="figures",
                        help="Directory for saved figures (default: figures/)")
    parser.add_argument("--fmt", default="png",
                        choices=["png", "pdf", "svg"],
                        help="Primary output format (PDF always saved too; default: png)")
    parser.add_argument("--plot-best-vs-count", action="store_true",
                        help="Plot best RMSE vs sensor count")
    parser.add_argument("--plot-heatmap", action="store_true",
                        help="Plot sparse 2-IMU heatmap from saved configs")
    parser.add_argument("--plot-bar", action="store_true",
                        help="Plot horizontal bar chart (all configurations)")
    parser.add_argument("--plot-all", action="store_true",
                        help="Generate all plots")
    args = parser.parse_args()

    # Resolve flags
    do_best    = args.plot_best_vs_count or args.plot_all
    do_heatmap = args.plot_heatmap       or args.plot_all
    do_bar     = args.plot_bar           or args.plot_all

    # Default: best-vs-count + heatmap if no flag given
    if not any([args.plot_best_vs_count, args.plot_heatmap,
                args.plot_bar, args.plot_all]):
        do_best    = True
        do_heatmap = True

    # Load data
    print(f"\n  Reading {args.input_csv} ...")
    all_rows = load_all_configs(args.input_csv)
    print(f"    {len(all_rows)} configurations loaded.")

    print(f"  Reading {args.best_csv} ...")
    best_rows = load_best_by_count(args.best_csv)
    print(f"    {len(best_rows)} best-per-count entries loaded.")

    meta = {}
    if os.path.exists(args.input_json):
        print(f"  Reading {args.input_json} ...")
        meta = load_json_meta(args.input_json)
        cfg  = meta.get("config", {})
        print(f"    Study config: alpha={cfg.get('alpha')}, "
              f"shapes={cfg.get('num_shapes')}, "
              f"noise_real={cfg.get('num_noise_realizations')}, "
              f"steps={cfg.get('steps')}, "
              f"meas_std={cfg.get('meas_std_deg')} deg")

    matplotlib.rcParams.update({
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    print(f"\n  Output directory: {args.output_dir}/\n")

    if do_best:
        print("  [A] Plotting best RMSE vs sensor count ...")
        plot_best_vs_count(best_rows, args.output_dir, args.fmt)

    if do_heatmap:
        print("  [B] Plotting 2-IMU sparse heatmap ...")
        plot_heatmap_2imu(all_rows, args.output_dir, args.fmt)

    if do_bar:
        print("  [C] Plotting bar chart (all configs) ...")
        plot_bar_all_configs(all_rows, args.output_dir, args.fmt)

    print("\n  Done.")


if __name__ == "__main__":
    main()
