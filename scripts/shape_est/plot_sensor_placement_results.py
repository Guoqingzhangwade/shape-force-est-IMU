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
  A1. best_vs_count_full.png/.pdf   -- best RMSE vs count, all 1-5 IMUs
  A2. best_vs_count_main.png/.pdf   -- main manuscript version, 2-5 IMUs only
  B1. heatmap_2imu_annotated.png/.pdf -- 2-IMU heatmap with cell values
  B2. heatmap_2imu_clean.png/.pdf     -- 2-IMU heatmap without cell values
  C.  bar_all_configs.png/.pdf        -- horizontal bar chart (optional)

Heatmap note
------------
The saved CSV contains the 6 predefined 2-IMU configs (positions {0.25, 0.50,
0.75, 1.00}), not a full grid sweep.  The heatmap is reconstructed as a sparse
4x4 grid with symmetric fill and masked diagonal.

Axis-centering fix: pcolormesh is called with cell-EDGE coordinates computed
from the sensor-position centers.  Ticks are then placed at the center values,
so labels (e.g. "0.25") appear at the CENTER of each cell.

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
#  FIGURE A: BEST RMSE vs SENSOR COUNT  (two versions)
# ============================================================

def _draw_best_vs_count(ax: plt.Axes, best_rows: List[Dict],
                        annotate: bool = True) -> None:
    """
    Shared drawing logic for best-vs-count.  `best_rows` may be a subset
    (e.g. only 2-5 IMUs for the manuscript version).
    """
    counts = [r["n_sensors"]       for r in best_rows]
    means  = [r["rmse_final_mean"] for r in best_rows]
    stds   = [r["rmse_final_std"]  for r in best_rows]
    colors = [_COUNT_COLOR.get(n, "gray") for n in counts]

    # Connecting line
    ax.plot(counts, means, "-", color="#888888",
            linewidth=LW, zorder=2, alpha=0.6)

    # Error bars
    ax.errorbar(counts, means, yerr=stds,
                fmt="none", ecolor="#555555",
                capsize=5, capthick=1.5, elinewidth=1.2, zorder=3)

    # Colored markers
    for n, m, c in zip(counts, means, colors):
        ax.scatter([n], [m], color=c, s=MS ** 2, zorder=4,
                   edgecolors="white", linewidths=1.0)

    if annotate:
        # Alternate labels above / below every other point to avoid crowding.
        # Use axes-fraction x so labels always sit just right of the last tick,
        # independent of data scale.
        for k, row in enumerate(best_rows):
            n   = row["n_sensors"]
            m   = row["rmse_final_mean"]
            pos = row["imu_positions"]
            lbl = "{" + ", ".join(f"{v:.2f}" for v in pos) + "}"
            dy  = 7 if k % 2 == 0 else -14   # alternating vertical offset
            ax.annotate(
                lbl,
                xy=(n, m),
                xytext=(10, dy),
                textcoords="offset points",
                fontsize=9,
                color="#333333",
                va="bottom" if dy > 0 else "top",
                arrowprops=dict(arrowstyle="-",
                                color="#bbbbbb",
                                lw=0.7,
                                shrinkA=4, shrinkB=2),
            )

    ax.set_xticks(counts)
    ax.set_xticklabels([str(n) for n in counts], fontsize=FS_TICK)
    ax.set_xlabel("Number of IMUs", fontsize=FS_LABEL)
    ax.set_ylabel("Best final RMSE\n(modal coefficients)", fontsize=FS_LABEL,
                  labelpad=8)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")


def plot_best_vs_count(best_rows: List[Dict],
                       output_dir: str,
                       fmt: str = "png") -> None:
    """
    Generate two versions of the best-RMSE-vs-count figure:
      A1. Full  (1-5 IMUs)  -> best_vs_count_full
      A2. Main  (2-5 IMUs)  -> best_vs_count_main  (manuscript figure)
    """
    best_rows = sorted(best_rows, key=lambda r: r["n_sensors"])

    # ── A1: Full (1–5 IMUs) ───────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    _draw_best_vs_count(ax1, best_rows, annotate=True)
    fig1.tight_layout()
    _save(fig1, output_dir, "best_vs_count_full", fmt)
    plt.close(fig1)

    # ── A2: Main manuscript (2–5 IMUs only) ──────────────────────────────
    main_rows = [r for r in best_rows if r["n_sensors"] >= 2]
    means2    = [r["rmse_final_mean"] for r in main_rows]
    stds2     = [r["rmse_final_std"]  for r in main_rows]

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    _draw_best_vs_count(ax2, main_rows, annotate=True)

    # y-axis: give 30% headroom above and 20% below for error bars + labels
    y_top = max(m + s for m, s in zip(means2, stds2)) * 1.30
    y_bot = max(min(m - s for m, s in zip(means2, stds2)) * 0.80, 0)
    ax2.set_ylim(y_bot, y_top)

    # Extra left margin so the two-line y-label is never clipped
    fig2.subplots_adjust(left=0.16, right=0.88, top=0.93, bottom=0.13)
    _save(fig2, output_dir, "best_vs_count_main", fmt)
    plt.close(fig2)

    print("  [A] best_vs_count_full and best_vs_count_main saved.")


# ============================================================
#  FIGURE B: SPARSE 2-IMU HEATMAP  (two versions)
# ============================================================

def _build_heatmap_data(all_rows: List[Dict]):
    """
    Extract 2-IMU rows, build grid, edge coordinates, and mask.
    Returns (positions, edges, grid, diag_mask, vmin, vmax) or None.

    Axis-centering explanation
    --------------------------
    pcolormesh(X_edges, Y_edges, Z) draws cell (i,j) as the rectangle
    [X_edges[j], X_edges[j+1]] x [Y_edges[i], Y_edges[i+1]].
    To make the tick label at position p land at the CENTER of its cell,
    the edges must bracket p symmetrically:
        edges[k]   = midpoint of p[k-1] and p[k]   (for k = 1 .. N-1)
        edges[0]   = p[0]  - half the first gap
        edges[N]   = p[-1] + half the last gap
    Then ax.set_xticks(p) / ax.set_yticks(p) places labels at cell centers.
    """
    two_imu = [r for r in all_rows if r["n_sensors"] == 2]
    if not two_imu:
        return None

    pos_set = set()
    for r in two_imu:
        for p in r["imu_positions"]:
            pos_set.add(round(p, 4))
    positions = sorted(pos_set)
    N         = len(positions)
    pos_idx   = {p: i for i, p in enumerate(positions)}

    grid = np.full((N, N), np.nan)
    for r in two_imu:
        p1, p2 = [round(v, 4) for v in r["imu_positions"]]
        i, j   = pos_idx[p1], pos_idx[p2]
        val    = r["rmse_final_mean"]
        grid[i, j] = val
        grid[j, i] = val     # symmetric

    diag_mask = np.zeros((N, N), dtype=bool)
    np.fill_diagonal(diag_mask, True)

    # Cell-edge coordinates (the centering fix)
    pos_arr    = np.array(positions)
    edges      = np.empty(N + 1)
    edges[0]    = pos_arr[0]  - (pos_arr[1]  - pos_arr[0])  / 2.0
    edges[1:-1] = (pos_arr[:-1] + pos_arr[1:]) / 2.0
    edges[-1]   = pos_arr[-1] + (pos_arr[-1] - pos_arr[-2]) / 2.0

    valid_vals  = grid[~diag_mask & ~np.isnan(grid)]
    vmin, vmax  = float(valid_vals.min()), float(valid_vals.max())

    return positions, edges, grid, diag_mask, vmin, vmax


def _draw_heatmap(ax: plt.Axes,
                  positions, edges, grid, diag_mask,
                  vmin: float, vmax: float,
                  annotate: bool) -> plt.cm.ScalarMappable:
    """
    Shared drawing logic for both heatmap variants.
    Returns the pcolormesh object for colorbar attachment.
    """
    N = len(positions)

    cmap = plt.cm.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color="#cccccc")    # diagonal / NaN -> light grey

    grid_masked = np.ma.array(grid, mask=diag_mask | np.isnan(grid))

    pcm = ax.pcolormesh(edges, edges, grid_masked,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        shading="flat")

    # Diagonal: hatched grey rectangle
    for k in range(N):
        rect = mpatches.Rectangle(
            (edges[k], edges[k]), edges[k+1] - edges[k], edges[k+1] - edges[k],
            linewidth=0, facecolor="#999999", hatch="///",
            alpha=0.55, zorder=3
        )
        ax.add_patch(rect)

    # Optional: RMSE value inside each valid cell
    if annotate:
        for i in range(N):
            for j in range(N):
                if i == j or np.isnan(grid[i, j]):
                    continue
                cx = (edges[j] + edges[j+1]) / 2.0
                cy = (edges[i] + edges[i+1]) / 2.0
                # Choose text color for contrast against colormap
                norm_val = (grid[i, j] - vmin) / max(vmax - vmin, 1e-12)
                txt_color = "white" if norm_val > 0.65 else "black"
                ax.text(cx, cy, f"{grid[i, j]:.4f}",
                        ha="center", va="center",
                        fontsize=9, color=txt_color,
                        fontweight="bold", zorder=5)

    # Thin white cell-boundary lines (only between cells, not at outer edges)
    for e in edges[1:-1]:
        ax.axhline(e, color="white", linewidth=0.8, zorder=2)
        ax.axvline(e, color="white", linewidth=0.8, zorder=2)

    # Axes — ticks at sensor position CENTERS (the key alignment step)
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels([f"{p:.2f}" for p in positions], fontsize=FS_TICK)
    ax.set_yticklabels([f"{p:.2f}" for p in positions], fontsize=FS_TICK)
    ax.set_xlabel("Sensor 2 position $s_2$", fontsize=FS_LABEL, labelpad=8)
    ax.set_ylabel("Sensor 1 position $s_1$", fontsize=FS_LABEL, labelpad=8)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(edges[0], edges[-1])
    ax.set_aspect("equal")

    return pcm


def plot_heatmap_2imu(all_rows: List[Dict],
                      output_dir: str,
                      fmt: str = "png") -> None:
    """
    Generate two heatmap variants from the 6 predefined 2-IMU configs:
      B1. heatmap_2imu_annotated  -- RMSE values printed inside cells
      B2. heatmap_2imu_clean      -- no cell annotations (cleaner for paper)
    """
    data = _build_heatmap_data(all_rows)
    if data is None:
        print("  [heatmap skipped] No 2-IMU data found in CSV.")
        return

    positions, edges, grid, diag_mask, vmin, vmax = data
    n_data = sum(1 for r in all_rows if r["n_sensors"] == 2)

    for annotate, stem in [(True,  "heatmap_2imu_annotated"),
                           (False, "heatmap_2imu_clean")]:
        # constrained_layout handles colorbar + label spacing automatically
        fig, ax = plt.subplots(figsize=(6.2, 5.2),
                               layout="constrained")
        pcm = _draw_heatmap(ax, positions, edges, grid, diag_mask,
                             vmin, vmax, annotate=annotate)

        cb = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046, aspect=20)
        cb.set_label("Mean final RMSE", fontsize=FS_LEGEND + 1)
        cb.ax.tick_params(labelsize=FS_TICK - 1)

        _save(fig, output_dir, stem, fmt)
        plt.close(fig)

    print(f"  [B] heatmap_2imu_annotated and heatmap_2imu_clean saved.")
    print(f"      ({n_data} predefined 2-IMU configurations; "
          "positions = " + str([round(p, 2) for p in positions]) + ")")
    print("  For a dense grid heatmap re-run:")
    print("    python sensor_placement_study_final.py --heatmap-grid 8")


# ============================================================
#  FIGURE C: HORIZONTAL BAR CHART (all configs)
# ============================================================

def _config_label(row: Dict) -> str:
    """Convert config name to use curly brackets for sensor layout."""
    n   = row["n_sensors"]
    pos = row["imu_positions"]
    return f"{n}-IMU " + "{" + ", ".join(f"{v:.2f}" for v in pos) + "}"


def plot_bar_all_configs(all_rows: List[Dict],
                         output_dir: str,
                         fmt: str = "png") -> None:
    """Horizontal bar chart of final RMSE for all configurations."""
    names  = [_config_label(r)     for r in all_rows]
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
    plt.close(fig)


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
        print("  [A] Plotting best RMSE vs sensor count (full + main) ...")
        plot_best_vs_count(best_rows, args.output_dir, args.fmt)

    if do_heatmap:
        print("  [B] Plotting 2-IMU heatmap (annotated + clean) ...")
        plot_heatmap_2imu(all_rows, args.output_dir, args.fmt)

    if do_bar:
        print("  [C] Plotting bar chart (all configs) ...")
        plot_bar_all_configs(all_rows, args.output_dir, args.fmt)

    print("\n  Done.")
    print(f"  All figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
