#!/usr/bin/env python3
"""
Plot-only companion for constrained tip-wrench estimation results.

Reads constrained_results.csv (from evaluate_kirchhoff_wrench_constrained.py)
and generates:

  Figure 1 — 3-row violin grid
    Rows    : constrained cases (force_only, transverse_force, moment_only)
    Columns : error type (force error [mN] | moment error [mN·m])
    Violins : oracle_constrained, ekf_unconstrained, ekf_constrained
              (one violin per method, merged over layouts; or split by layout)

  Figure 2 — NRMSE bar chart
    Side-by-side bars per (case, method, layout)
    Separate panels for force NRMSE and moment NRMSE

  Figure 3 — Direction-error violin
    Force direction error and moment direction error

  Figure 4 — Constrained vs unconstrained improvement heatmap
    Relative reduction in force / moment error from EKF-unconstrained
    to EKF-constrained, broken down by case and layout

Usage
-----
  cd src/shape_force_est_imu/crt
  python plot_kirchhoff_wrench_constrained_summary.py
  python plot_kirchhoff_wrench_constrained_summary.py \\
      --results gt_data_constrained/results/constrained_results.csv \\
      --save-dir gt_data_constrained/results
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
LABEL_FS  = 12
TICK_FS   = 10
TITLE_FS  = 12
LEGEND_FS = 10
ANN_FS    = 8.5
LW        = 1.5

_CASE_LABELS = {
    "force_only"       : "Force only",
    "transverse_force" : "Transverse force",
    "moment_only"      : "Moment only",
}
_CASE_ORDER = ["force_only", "transverse_force", "moment_only"]

_METHOD_COLORS = {
    "oracle_constrained" : "#2ca02c",   # green
    "ekf_unconstrained"  : "#ff7f0e",   # orange
    "ekf_constrained"    : "#1f77b4",   # blue
}
_METHOD_LABELS = {
    "oracle_constrained" : "Oracle (constrained)",
    "ekf_unconstrained"  : "EKF unconstrained",
    "ekf_constrained"    : "EKF constrained",
}
_METHOD_ORDER = ["oracle_constrained", "ekf_unconstrained", "ekf_constrained"]
_LAYOUT_ORDER = ["GT", "2-IMU", "3-IMU"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def cast_floats(rows: list[dict], keys: list[str]) -> list[dict]:
    for r in rows:
        for k in keys:
            if k in r and r[k] not in ("", None):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    r[k] = float("nan")
    return rows


def extract(
    rows: list[dict],
    ccase: str,
    method: str,
    layout: str,
    key: str,
) -> np.ndarray:
    vals = [r[key] for r in rows
            if r.get("constrained_case") == ccase
            and r.get("method") == method
            and r.get("layout_name") == layout]
    return np.array([float(v) for v in vals if v not in ("", None, "nan")])


# ---------------------------------------------------------------------------
# Violin helper
# ---------------------------------------------------------------------------

def _violin(ax, pos: float, data: np.ndarray, color: str, width: float = 0.55):
    if len(data) == 0:
        return
    vp = ax.violinplot(data, positions=[pos], widths=width,
                       showmedians=True, showextrema=False)
    for pc in vp["bodies"]:
        pc.set_facecolor(color); pc.set_alpha(0.50); pc.set_edgecolor(color)
    vp["cmedians"].set_color(color); vp["cmedians"].set_linewidth(2.0)
    ax.scatter([pos], [np.mean(data)], marker="D", s=22, color=color, zorder=5)
    return float(np.mean(data))


# ---------------------------------------------------------------------------
# Figure 1: 3-row × 2-column violin grid
# ---------------------------------------------------------------------------

def fig_violin_grid(rows: list[dict], save_stem: Path) -> None:
    """
    Rows    : constrained case (force_only, transverse, moment)
    Columns : force error [mN] | moment error [mN·m]
    Violins : oracle_constrained, ekf_unconstrained, ekf_constrained
              — pooled across layouts for clarity (layout split in Fig 2)
    """
    fig, axes = plt.subplots(
        3, 2, figsize=(10, 9), sharey="row",
        gridspec_kw={"hspace": 0.42, "wspace": 0.22},
    )
    fig.suptitle("Constrained Tip-Wrench Estimation Errors",
                 fontsize=TITLE_FS + 2, fontweight="bold", y=1.01)

    ylabels_force  = ["Force error [mN]"] * 3
    ylabels_moment = ["Moment error [mN·m]"] * 3

    for row_idx, ccase in enumerate(_CASE_ORDER):
        for col_idx, (metric_key, scale, ylabel) in enumerate([
            ("force_err_N",   1e3, ylabels_force[row_idx]),
            ("moment_err_Nm", 1e3, ylabels_moment[row_idx]),
        ]):
            ax = axes[row_idx][col_idx]

            # Pool layouts: oracle has only "GT", EKF methods have "2-IMU" + "3-IMU"
            positions_map = {
                "oracle_constrained": (1, "GT"),
                "ekf_unconstrained" : None,   # merged below
                "ekf_constrained"   : None,
            }
            pos = 1
            tick_pos, tick_lbl = [], []

            # Oracle
            data = extract(rows, ccase, "oracle_constrained", "GT", metric_key) * scale
            mean = _violin(ax, pos, data, _METHOD_COLORS["oracle_constrained"])
            if mean is not None:
                ax.text(pos, mean * 1.05, f"{mean:.1f}", ha="center", va="bottom",
                        fontsize=ANN_FS, color=_METHOD_COLORS["oracle_constrained"],
                        fontweight="bold")
            tick_pos.append(pos); tick_lbl.append("Oracle")
            pos += 1

            for method in ["ekf_unconstrained", "ekf_constrained"]:
                # Pool 2-IMU and 3-IMU
                d2 = extract(rows, ccase, method, "2-IMU", metric_key) * scale
                d3 = extract(rows, ccase, method, "3-IMU", metric_key) * scale
                data = np.concatenate([d2, d3])
                mean = _violin(ax, pos, data, _METHOD_COLORS[method])
                if mean is not None:
                    ax.text(pos, mean * 1.05, f"{mean:.1f}", ha="center", va="bottom",
                            fontsize=ANN_FS, color=_METHOD_COLORS[method],
                            fontweight="bold")
                short = "Unc." if "unc" in method else "Con."
                tick_pos.append(pos); tick_lbl.append(f"EKF\n{short}")
                pos += 1

            if row_idx == 0:
                col_titles = ["Force error [mN]", "Moment error [mN·m]"]
                ax.set_title(col_titles[col_idx], fontsize=TITLE_FS, fontweight="bold")

            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbl, fontsize=TICK_FS)
            ax.tick_params(axis="y", labelsize=TICK_FS)
            ax.set_ylabel(f"{_CASE_LABELS[ccase]}\n{ylabel}", fontsize=LABEL_FS - 1)
            ax.set_xlim(0.4, pos - 0.4)
            ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
            ax.set_axisbelow(True)

    handles = [mpatches.Patch(facecolor=_METHOD_COLORS[m], label=_METHOD_LABELS[m],
                               alpha=0.7) for m in _METHOD_ORDER]
    fig.legend(handles=handles, fontsize=LEGEND_FS, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = save_stem.parent / (save_stem.name + "_violin." + ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: NRMSE bar chart (layout-split)
# ---------------------------------------------------------------------------

def fig_nrmse_bars(rows: list[dict], save_stem: Path) -> None:
    """Side-by-side NRMSE bars per (method, layout) for each constrained case."""
    combos = [
        ("oracle_constrained", "GT",    _METHOD_COLORS["oracle_constrained"], "Oracle"),
        ("ekf_unconstrained",  "2-IMU", _METHOD_COLORS["ekf_unconstrained"],  "EKF Unc. 2-IMU"),
        ("ekf_unconstrained",  "3-IMU", "#d62728",                            "EKF Unc. 3-IMU"),
        ("ekf_constrained",    "2-IMU", _METHOD_COLORS["ekf_constrained"],    "EKF Con. 2-IMU"),
        ("ekf_constrained",    "3-IMU", "#17becf",                            "EKF Con. 3-IMU"),
    ]
    n_cases  = len(_CASE_ORDER)
    n_combos = len(combos)
    x        = np.arange(n_cases)
    bar_w    = 0.14
    offsets  = np.linspace(-(n_combos - 1) / 2, (n_combos - 1) / 2, n_combos) * bar_w

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Constrained Wrench Estimation: NRMSE by Case and Method",
                 fontsize=TITLE_FS + 1, fontweight="bold")

    for ax, mk, ylabel in [
        (axes[0], "nrmse_force",  "Force NRMSE [%]"),
        (axes[1], "nrmse_moment", "Moment NRMSE [%]"),
    ]:
        for (method, layout, color, lbl), off in zip(combos, offsets):
            means = []
            for ccase in _CASE_ORDER:
                vals = extract(rows, ccase, method, layout, mk) * 100
                means.append(float(np.mean(vals)) if len(vals) else 0.0)

            bars = ax.bar(x + off, means, bar_w, color=color, alpha=0.82,
                          edgecolor="k", linewidth=0.5, label=lbl)
            for bar, mean in zip(bars, means):
                if mean > 2:
                    ax.text(bar.get_x() + bar.get_width() / 2, mean * 1.02,
                            f"{mean:.0f}%", ha="center", va="bottom",
                            fontsize=ANN_FS - 1.5, fontweight="bold", color=color)

        ax.set_xticks(x)
        ax.set_xticklabels([_CASE_LABELS[c] for c in _CASE_ORDER],
                           fontsize=TICK_FS, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    axes[1].legend(fontsize=LEGEND_FS - 1, loc="upper right", ncol=1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = save_stem.parent / (save_stem.name + "_nrmse." + ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Direction error violin
# ---------------------------------------------------------------------------

def fig_direction_errors(rows: list[dict], save_stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Constrained Estimation: Direction Errors",
                 fontsize=TITLE_FS + 1, fontweight="bold")

    for ax, key, ylabel in [
        (axes[0], "force_dir_err_deg",  "Force direction error [°]"),
        (axes[1], "moment_dir_err_deg", "Moment direction error [°]"),
    ]:
        pos = 1
        tick_pos, tick_lbl = [], []

        for ci, ccase in enumerate(_CASE_ORDER):
            # group center label
            group_start = pos
            for method in _METHOD_ORDER:
                layouts = ["GT"] if method == "oracle_constrained" else ["2-IMU", "3-IMU"]
                for layout in layouts:
                    data = extract(rows, ccase, method, layout, key)
                    if len(data):
                        _violin(ax, pos, data, _METHOD_COLORS[method], width=0.45)
                    pos += 1
            # group tick at center
            tick_pos.append((group_start + pos - 1) / 2)
            tick_lbl.append(_CASE_LABELS[ccase])
            pos += 0.6   # gap between groups

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=TICK_FS, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    handles = [mpatches.Patch(facecolor=_METHOD_COLORS[m], label=_METHOD_LABELS[m],
                               alpha=0.7) for m in _METHOD_ORDER]
    axes[1].legend(handles=handles, fontsize=LEGEND_FS)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = save_stem.parent / (save_stem.name + "_direction." + ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Improvement ratio heatmap (constrained vs unconstrained)
# ---------------------------------------------------------------------------

def fig_improvement_heatmap(rows: list[dict], save_stem: Path) -> None:
    """
    For EKF runs, compute relative improvement:
      improvement = (err_unconstrained - err_constrained) / err_unconstrained × 100%
    Positive = constrained is better.
    """
    layouts = ["2-IMU", "3-IMU"]
    metrics = [("force_err_N", "Force error"), ("moment_err_Nm", "Moment error")]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Relative Improvement: EKF Constrained vs Unconstrained [%]\n"
                 "(positive = constrained is better)",
                 fontsize=TITLE_FS, fontweight="bold")

    for ax, (mk, title) in zip(axes, metrics):
        data_mat = np.zeros((len(_CASE_ORDER), len(layouts)))
        for ci, ccase in enumerate(_CASE_ORDER):
            for li, layout in enumerate(layouts):
                unc = extract(rows, ccase, "ekf_unconstrained", layout, mk)
                con = extract(rows, ccase, "ekf_constrained",   layout, mk)
                if len(unc) and len(con):
                    mean_unc = float(np.mean(unc))
                    mean_con = float(np.mean(con))
                    if mean_unc > 1e-12:
                        data_mat[ci, li] = (mean_unc - mean_con) / mean_unc * 100
                    else:
                        data_mat[ci, li] = 0.0

        im = ax.imshow(data_mat, cmap="RdYlGn", vmin=-30, vmax=60, aspect="auto")
        ax.set_xticks(range(len(layouts)))
        ax.set_xticklabels(layouts, fontsize=TICK_FS)
        ax.set_yticks(range(len(_CASE_ORDER)))
        ax.set_yticklabels([_CASE_LABELS[c] for c in _CASE_ORDER], fontsize=TICK_FS)
        ax.set_title(title, fontsize=TITLE_FS)
        for ci in range(len(_CASE_ORDER)):
            for li in range(len(layouts)):
                val = data_mat[ci, li]
                ax.text(li, ci, f"{val:+.1f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if abs(val) > 25 else "black")
        plt.colorbar(im, ax=ax, label="improvement [%]", shrink=0.8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = save_stem.parent / (save_stem.name + "_improvement." + ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"  Saved -> {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot constrained wrench estimation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        default="gt_data_constrained/results/constrained_results.csv",
        help="CSV from evaluate_kirchhoff_wrench_constrained.py",
    )
    parser.add_argument(
        "--save-dir",
        default="gt_data_constrained/results",
        help="output folder for figures",
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
    save_stem = save_dir / "constrained_wrench_est"

    float_keys = [
        "force_err_N", "moment_err_Nm",
        "force_dir_err_deg", "moment_dir_err_deg",
        "nrmse_force", "nrmse_moment",
        "force_gt_norm_N", "moment_gt_norm_Nm",
    ]

    print(f"Loading {csv_path} …")
    rows = load_csv(csv_path)
    rows = cast_floats(rows, float_keys)

    n_oracle = len([r for r in rows if "oracle" in r.get("method", "")])
    n_ekf    = len([r for r in rows if "ekf"    in r.get("method", "")])
    print(f"  {len(rows)} total rows  (oracle={n_oracle}, ekf={n_ekf})")

    # Quick console summary
    print("\nMean errors by (case, method, layout):")
    _case_order   = ["force_only", "transverse_force", "moment_only"]
    _method_order = ["oracle_constrained", "ekf_unconstrained", "ekf_constrained"]
    for ccase in _case_order:
        for method in _method_order:
            layouts = (["GT"] if method == "oracle_constrained"
                       else ["2-IMU", "3-IMU"])
            for layout in layouts:
                fe = extract(rows, ccase, method, layout, "force_err_N")  * 1e3
                me = extract(rows, ccase, method, layout, "moment_err_Nm") * 1e3
                nf = extract(rows, ccase, method, layout, "nrmse_force")  * 100
                if len(fe):
                    print(f"  {ccase:<20} {method:<22} {layout:<8}  "
                          f"F={np.mean(fe):6.2f} mN  M={np.mean(me):7.3f} mN·m  "
                          f"NRMSE-F={np.mean(nf):5.1f}%")

    print("\nGenerating figures …")
    fig_violin_grid(rows, save_stem)
    fig_nrmse_bars(rows, save_stem)
    fig_direction_errors(rows, save_stem)
    fig_improvement_heatmap(rows, save_stem)
    print("Done.")


if __name__ == "__main__":
    main()
